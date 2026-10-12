# XAI: Integrated Gradients (원자 임베딩 기준, 기준점 0, 32단계) — 누수 없음·우리 방식, 시드 0 모델
# 쌍 모델: Δ̂(A→B)에 대한 A·B 원자 기여 / 분자별 모델: Δ̂ = f(B) − f(A) → B는 f(B) 기여, A는 −f(A) 기여
# Δ에 대한 순 기여: 공통 부분(MCS) 원자는 A의 원자와 B의 짝 원자 기여를 더해 하나로 본다 (분자별 모델은 A·B 기여가 크게 상쇄되므로)
# 바뀐 부분 집중도 = 바뀐 원자 |기여| 합 ÷ (바뀐 원자 |기여| 합 + 공통 원자 짝 |순 기여| 합). 우연 기대값 = 바뀐 원자 수 ÷ (바뀐 원자 수 + 공통 원자 짝 수)
import torch, numpy as np, pandas as pd, time
STEPS, N_EACH = 256, 1000   # 256단계: 32·128단계는 완전성 오차가 컸다
XAI_OUT = '/home/bising0922/Capstone/runs/tmp/xai'

def emb_modules(model):
    return [m for n, m in model.named_modules() if n.endswith('.inp') or n.endswith('x_embedding1') or n.endswith('x_embedding2')]

def ig(model, call, graphs_ab):
    """call(): 복사본(STEPS개)별 출력. graphs_ab: {'A': 묶음, 'B': 묶음} → {'A': 원자별 기여, 'B': ...}"""
    alphas = (torch.arange(STEPS, device=DEVICE).float() + 0.5) / STEPS   # 중점 리만 합
    ptr = {k: g.x.untyped_storage().data_ptr() for k, g in graphs_ab.items()}
    store = []
    def hook(mod, inp, out):
        n = out.shape[0] // STEPS
        scaled = out * alphas.repeat_interleave(n).view(-1, 1)
        scaled.retain_grad()
        which = next(k for k, p in ptr.items() if inp[0].untyped_storage().data_ptr() == p)
        store.append((which, out.detach(), scaled, n))
        return scaled
    hs = [m.register_forward_hook(hook) for m in emb_modules(model)]
    try:
        model.zero_grad(); call().sum().backward()
    finally:
        for h in hs: h.remove()
    res = {}
    for which, e, s, n in store:
        a = (e * s.grad).sum(-1).view(STEPS, n).mean(0).detach().cpu().numpy()
        res[which] = res.get(which, 0) + a
    return res

def at_scale(model, call, scale):
    """원자 임베딩을 scale배 한 상태의 출력 (scale=0 → IG 기준점)"""
    hs = [m.register_forward_hook(lambda mod, inp, out: out * scale) for m in emb_modules(model)]
    try:
        with torch.no_grad(): return call().detach().cpu().numpy()
    finally:
        for h in hs: h.remove()

from rdkit.Chem import rdFMCS
def common_map(r):
    """공통 부분 원자 짝 (A 번호, B 번호). 바뀐 원자 집합과 맞는 대응을 고른다"""
    a, b = Chem.MolFromSmiles(r.smiles_a), Chem.MolFromSmiles(r.smiles_b)
    q = Chem.MolFromSmarts(rdFMCS.FindMCS([a, b], atomCompare=rdFMCS.AtomCompare.CompareElements, bondCompare=rdFMCS.BondCompare.CompareOrder,
                                          ringMatchesRingOnly=True, timeout=MCS_TIMEOUT).smartsString)
    ma = a.GetSubstructMatch(q); want = set(r.atoms_b)
    mbs = b.GetSubstructMatches(q, uniquify=False, maxMatches=1000)
    mb = next((m for m in mbs if set(range(b.GetNumAtoms())) - set(m) == want), mbs[0] if mbs else ())
    return list(zip(ma, mb))

def load(model, path):
    model.load_state_dict(torch.load(path, map_location=DEVICE)); return model.to(DEVICE).eval()

MODELS = {
    'GINE 쌍': ('pair', load(PairModel(), f'{RUNS}/gnn_a0_{TAGS["all"]}_seed0.pt'), graphs),
    'GINE 분자별': ('mol', load(MolModel(), f'{RUNS}/gnnmol_{MOL_TAG}_seed0.pt'), graphs),
    'Mole-BERT 쌍': ('pair', load(PTPairModel(), f'{RUNS}/gnn_a0_{PT_TAGS["all"]}_seed0.pt'), PT_GRAPHS),
    'Mole-BERT 분자별': ('mol', load(PTMolModel(), f'{RUNS}/gnnmol_{PTMOL_TAG}_seed0.pt'), PT_GRAPHS),
}
ev = pairs_all[(pairs_all.split == 'test') & (pairs_all.reason == 'pass')]
SAMPLE = pd.concat([ev[ev.is_cliff].sample(N_EACH, random_state=SEED), ev[~ev.is_cliff].sample(N_EACH, random_state=SEED)]).reset_index(drop=True)
rows, atoms = [], []
t0 = time.time()
for k, r in enumerate(SAMPLE.itertuples()):
    rdf = SAMPLE.iloc[[k]]
    tgt = torch.full((STEPS,), DEV_TARGETS.index(r.target), dtype=torch.long, device=DEVICE)
    pairs_map = common_map(r)
    for name, (kind, model, gd) in MODELS.items():
        if kind == 'pair':
            ga, gb, e, t, _ = collate(samples(rdf, gdict=gd) * STEPS)
            ga, gb, e = ga.to(DEVICE), gb.to(DEVICE), e.to(DEVICE)
            res = ig(model, lambda: model(ga, gb, e, tgt), {'A': ga, 'B': gb})
            with torch.no_grad(): pred = float(model(ga, gb, e, tgt)[0])
            base = float(at_scale(model, lambda: model(ga, gb, e, tgt), 0.0)[0])
            aA, aB = res['A'], res['B']
        else:
            gA = Batch.from_data_list([gd[r.target, r.i]] * STEPS).to(DEVICE)
            gB = Batch.from_data_list([gd[r.target, r.j]] * STEPS).to(DEVICE)
            aB = ig(model, lambda: model(gB, tgt), {'B': gB})['B']
            aA = -ig(model, lambda: model(gA, tgt), {'A': gA})['A']
            with torch.no_grad(): pred = float(model(gB, tgt)[0] - model(gA, tgt)[0])
            base = float(at_scale(model, lambda: model(gB, tgt), 0.0)[0] - at_scale(model, lambda: model(gA, tgt), 0.0)[0])
        ch = np.abs(aA[list(r.atoms_a)]).sum() + np.abs(aB[list(r.atoms_b)]).sum()
        com = np.abs(np.array([aA[i] + aB[j] for i, j in pairs_map]) if pairs_map else np.zeros(1)).sum()
        n_ch = len(r.atoms_a) + len(r.atoms_b)
        rows.append(dict(k=k, model=name, is_cliff=r.is_cliff, delta=r.delta, pred=pred, base=base, attr_sum=float(aA.sum() + aB.sum()),
                         share=float(ch / (ch + com)), expected=float(n_ch / (n_ch + len(pairs_map))), n_changed=n_ch, n_common=len(pairs_map)))
        atoms.append(dict(k=k, model=name, attr_a=aA, attr_b=aB))
    if k % 200 == 0:
        print(f'{k}/{len(SAMPLE)}  {time.time() - t0:.0f}초', flush=True)
XAI = pd.DataFrame(rows)
XAI.to_pickle(f'{XAI_OUT}/xai_rows.pkl'); pd.DataFrame(atoms).to_pickle(f'{XAI_OUT}/xai_atoms.pkl')
SAMPLE.to_pickle(f'{XAI_OUT}/xai_sample.pkl')
# 완전성 확인: 기여 합 ≈ 예측 − 기준점(임베딩 0) 출력. 쌍 모델은 기준점 출력이 0이 아닐 수 있어 상관으로 본다
gap = (XAI.attr_sum - (XAI.pred - XAI.base)).abs()   # 완전성: 기여 합 = 예측 − 기준점 출력 이어야 한다
print('완전성 오차 (|기여 합 − (예측 − 기준점)|) 중앙값·95%:', XAI.assign(gap=gap).groupby('model').gap.quantile([0.5, 0.95]).round(4).to_dict())
print('예측 크기 중앙값', XAI.pred.abs().median().round(3))
s = XAI.assign(ratio=XAI.share / XAI.expected).groupby(['model', 'is_cliff'])[['share', 'expected', 'ratio']].mean().round(3)
print(s)
