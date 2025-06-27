# fitclo_web 
한국 폴리텍대학 대전 캠퍼스 인공지능과에서 2025년도에 시작된 AI(diffision모델)기반 온라인 가상피팅 프로젝트 입니다.


# 업데이트 예정 목록

모델의 최적화/경량화

모델의 추가학습 

의상의 카테고리 자동 생성

신체 정보 자동 생성

유저별 의상 추천 알고리즘

실시간 피팅(오버레이방식으로 예정)



# 사용 방법
하단의 추가 파일들의 다운로드, 압축 해제 완료 후 requirements를 통해 패키지들을 다운로드하신 다음

fitclo_web\fitclo\VSViton_serverTest\WebServer2경로의 app.py를 작동시키면 됩니다.


# 추가 파일들

ckpt파일과 데이터셋파일들은 하단 링크를 통해 다운로드 하면 됩니다.

[ckpts파일 다운로드](https://drive.google.com/file/d/1cIRS4SfAXGBGQwVyEUSKfmQ1jskQGrLp/view?usp=sharing)

[dataset파일 다운로드](https://drive.google.com/file/d/1o_SN2t765aiIwe115Pe8dzE8zAhknIvi/view?usp=sharing)



# 파일 및 폴더 배치 안내
상단 링크를 통해  ckpts와 webserver_datasets압축파일을 받은 뒤 
아래 지침에 따라 압축 해제된 파일과 폴더를 올바른 위치에 배치해 주세요.

## 1. `ckpts` 폴더 배치

* 압축 해제한 `ckpts` 폴더의 내용을 아래 경로에 **덮어쓰세요.**
    ```
    fitclo/
    └── StableVITON-master3/
        └── ckpts/
            ├── model_file_1.pth
            └── model_file_2.pt
            └── ... (기타 ckpts 파일들)
    ```

## 2. `webserver_dataset_test` 폴더 배치

* 압축 해제한 `webserver_dataset_test` 폴더의 이름을 **`test`로 변경**한 후, 아래 경로에 **덮어쓰세요.**
    ```
    fitclo/
    └── VSViton-serverTest/
        └── WebServer2/
            └── static/
                └── dataset/
                    └── test/
                        ├── image_1.jpg
                        └── image_2.png
                        └── ... (기타 test dataset 파일들)
    ```

## 3. `selectedimage_` 폴더 배치

* 압축 해제한 `selectedimage_` 폴더의 이름을 **`selectedimage`로 변경**한 후, 아래 경로에 배치하세요.
    ```
    fitclo/
    └── VSViton-serverTest/
        └── WebServer2/
            └── static/
                └── selectedimage/
                    ├── item_1.png
                    └── item_2.jpg
                    └── ... (기타 selectedimage 파일들)
    ```




    

---
