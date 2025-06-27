# fitclo_web

github에 모델의 업로드를 시도했지만 시간이 너무 오래 걸려 ckpt파일과 데이터셋파일들은 외부 링크로 업로드중입니다.
업로드 예상 시간은 약 20분입니다.


추가될 링크를 통해  ckpts와 webserver_datasets압축파일을 받은 뒤 
ckpts파일은 stableviton-mastre3의  ckpts파일에 압축해제하시면 됩니다.
webserver_datasets의 압축파일 내 selectedimage_폴더를 VSViton_serverTest/Webserver2/static/selectedimage폴더에 덮어씌우고,
압축파일 내 webserver_dataset_test폴더는 VSViton_serverTest/Webserver2/dataset에 옮기신 후 폴더명을'test'로 변경하면 됩니다.

최종적으로 
fitclo - StableVITON-master3 - ckpts <- 압축해제한 ckpts폴더를 덮어쓰기
       - VSViton-serverTest  - WebServer2 - static  - dataset - test <- 압축해제한 webserver_dataset_test 폴더를 'test'로 변경후 덮어쓰기
                                                    - selectedimage <- 압축해제한 selectedimage_폴더를 'selectedimage'로 변경
  
