Capstone 재현용 캐시 (박윤정)

1. 저장소(pair-pipeline 브랜치)를 받은 뒤, 저장소 최상위 폴더에서:
   cp /home/bising0922/share/capstone_cache_2026-10-12.tar.gz .
   sha256sum -c /home/bising0922/share/capstone_cache_2026-10-12.sha256   # "OK"가 나오면 정상
   tar -xzf capstone_cache_2026-10-12.tar.gz
   → data/, runs/ 아래 원래 자리에 들어간다 (306개 파일, 약 1.4GB)
2. 이후 방법은 저장소의 experiments/README.md 를 따른다.
3. 들어 있는 파일 목록: capstone_cache_2026-10-12_files.txt
