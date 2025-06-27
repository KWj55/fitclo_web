# fitclo_web

github에 모델의 업로드를 시도했지만 시간이 너무 오래 걸려 ckpt파일과 데이터셋파일들은 외부 링크로 업로드중입니다.
업로드 예상 시간은 약 20분입니다.




# 파일 및 폴더 배치 안내
추가될 링크를 통해  ckpts와 webserver_datasets압축파일을 받은 뒤 
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
