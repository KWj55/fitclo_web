from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import threading
from pathlib import Path
import sys
import os
from os.path import join as opj
import subprocess
from PIL import Image
import torch
import torchvision.transforms as transforms
import uuid
from datetime import datetime, date, timedelta
import shutil
from typing import Optional
import mysql.connector
from mysql.connector import Error
import hashlib
import jwt
from jwt.exceptions import PyJWTError
import yaml
from omegaconf import OmegaConf
import cv2
import numpy as np
from contextlib import asynccontextmanager
from importlib import import_module
from torch.utils.data import DataLoader

# stableviton 모델 import를 위한 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))

# StableVITON 디렉토리
stableviton_dir = os.path.join(parent_dir, 'stableviton-master3')
print(f"stableviton 디렉토리: {stableviton_dir}")

# Python path에 필요한 디렉토리들 추가
if stableviton_dir not in sys.path:
    sys.path.insert(0, stableviton_dir)

print("프로그램 시작") 

from cldm.model import create_model, load_state_dict
from cldm.cldm import ControlLDM
from cldm.plms_hacked import PLMSSampler
from utils import tensor2img, preprocess_image


class Model(): 
    def __init__(self):
        #경로 설정
        self.model_config_path = 'D:/kwj/fitclo/stableviton-master3/configs/VITONHD.yaml'
        self.model_load_path = 'D:/kwj/fitclo/stableviton-master3/ckpts/VITONHD_PBE_pose.ckpt'
        self.data_root_dir = 'D:/kwj/fitclo/VSViton_serverTest/WebServer2/static/selectedimage'
        self.save_dir = 'D:/kwj/fitclo/VSViton_serverTest/WebServer2/static/result'
    
        # 모델 파라미터 설정
        self.denoise_steps = 50
        self.batch_size = 1
        self.repaint = True
        self.unpair = True
        self.img_H = 512
        self.img_W = 384
        self.eta = 0.0
        self.model = None

        # stableviton 모델 설정(설정파일(yaml)에 구조가 정의되어있음)
        self.config = OmegaConf.load(self.model_config_path)
        #모델의 img_H, img_W 설정
        #img_H, img_W는 모델의 입력 이미지 크기
        self.config.model.params.img_H = self.img_H
        self.config.model.params.img_W = self.img_W
        #모델 파라미터를 config에서 가져옴
        self.params = self.config.model.params

        #각종 변수 선언
        self.model = None
        self.dataset = None
        self.dataloader = None
        self.shape = (4, self.img_H // 8, self.img_W // 8)  # 잠재 공간의 크기 설정
        self.uc_cross = None  # 조건 없는(unconditional) 컨디셔닝
        self.sampler = None

    # 모델 생성
    def create_model(self):
        # config_path가 None인 경우, config를 직접 전달하여 모델을 생성
        self.model = create_model(config_path=None, config=self.config)
        # 체크포인트 로드
        load_cp = torch.load(self.model_load_path, map_location="cpu")
        # 체크포인트가 state_dict를 포함하는 경우 해당 값을 사용
        load_cp = load_cp["state_dict"] if "state_dict" in load_cp.keys() else load_cp

        # 생성된 모델에 가중치 적용
        self.model.load_state_dict(load_cp)
        # cuda설정 디바이스로 이동(보통 gpu로 옮기기 위해 사용)
        self.model = self.model.cuda()
        # 모델을 평가 모드로 설정
        self.model.eval()
        # sampler 초기화
        self.sampler = PLMSSampler(self.model)

    #데이터셋모듈(아마 데이터셋 폴더)에서 dataset_name에 해당하는 클래스를 불러옴
    def set_dataset(self):
        # dataset 모듈에서 config.dataset_name에 해당하는 클래스를 가져옴
        # config.dataset_name은 config 파일에 정의된 데이터셋 이름입니다.
        self.dataset = getattr(import_module("dataset"), self.config.dataset_name)(
            data_root_dir=self.data_root_dir,
            img_H=self.img_H,
            img_W=self.img_W,
            is_paired=not self.unpair,  # default는 false
            is_test=True,
            is_sorted=True
        )
        self.dataloader =  DataLoader(self.dataset, num_workers=4, shuffle=False, batch_size=self.batch_size, pin_memory=True)

    #모델 추론
    def inference(self):
        for batch_idx, batch in enumerate(self.dataloader):
            print(f"{batch_idx}/{len(self.dataloader)}")

            # 모델의 get_input 함수를 사용하여 배치 데이터로부터 잠재 벡터(z)와 컨디셔닝 정보(c)를 추출
            z, c = self.model.get_input(batch, self.params.first_stage_key)
            bs = z.shape[0]  # 현재 배치의 크기
            c_crossattn = c["c_crossattn"][0][:bs]  # 크로스 어텐션에 사용될 컨디셔닝 정보

            if c_crossattn.ndim == 4:  # 컨디셔닝 정보가 이미지 형태(4차원 텐서)일 경우
                #
                c_crossattn = self.model.get_learned_conditioning(c_crossattn)
                c["c_crossattn"] = [c_crossattn]

            # 조건 없는(unconditional) 컨디셔닝 생성
            uc_cross = self.model.get_unconditional_conditioning(bs)
            uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
            uc_full["first_stage_cond"] = c["first_stage_cond"]

            for k, v in batch.items():
                # 배치 내의 모든 텐서들을 GPU로 이동시킵니다.
                #텐서타입이 아니라면 제외
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda()
            self.sampler.model.batch = batch

            ts = torch.full((1,),999, device=z.device, dtype=torch.long) #노이즈 생성강도 999로(매우 강하게)설정
            start_code = self.model.q_sample(z, ts)  #잠재 공간에 노이즈 생성 
            
            #노이즈를 준 잠재공간(최종 피쳐맵으로 부터 복원)
            samples,_,_ = self.sampler.sample(
                self.denoise_steps,
                bs,
                self.shape,
                c,
                x_T = start_code,  # 샘플링을 시작할 초기 노이즈 코드
                verbose=False,  # 상세 로그 출력 여부
                eta=self.eta,  # 노이즈 제거 과정에서의 eta 값(default: 0.0)
                unconditional_conditioning=uc_full,
            )

            # 생성된 잠재 공간의 샘플을 VAE 디코더를 통해 실제 이미지로 복원
            x_samples = self.model.decode_first_stage(samples)

            for sample_idx, (x_sample, fn, cloth_fn) in enumerate(zip(x_samples, batch['img_fn'], batch["cloth_fn"])):
                x_sample_img = tensor2img(x_sample)

                if self.repaint:
                    # repaint 옵션이 활성화된 경우, 특정 영역을 원본 이미지로 복원
                    repaint_agn_img = np.uint8((batch["image"][sample_idx].cpu().numpy()+1)/2 * 255) # [0,255]
                    repaint_arg_mask_img = batch["agn_mask"][sample_idx].cpu().numpy()  # 0 or 1
                    x_sample_img = repaint_agn_img * repaint_arg_mask_img + x_sample_img * (1 - repaint_arg_mask_img)
                    x_sample_img = np.uint8(x_sample_img)

            #이미지 저장 경로 설정
            to_path = opj(self.save_dir, f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}.jpg")
            # OpenCV는 BGR 형식으로 저장하므로 RGB를 BGR로 변환하여 저장
            cv2.imwrite(to_path, x_sample_img[:, :, ::-1])  

model = Model()  # 모델 인스턴스 생성


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # 시작 시 실행될 코드
        print("Starting application initialization...")
        print(f"현재 스크립트 위치: {current_dir}")
        print(f"업로드 폴더 위치: {UPLOAD_FOLDER}")
        print(f"결과 폴더 위치: {RESULT_FOLDER}")

        # 디렉토리 생성
        print("Creating necessary directories...")
        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
        SELECTED_PERSON_FOLDER.mkdir(parents=True, exist_ok=True)
        SELECTED_CLOTH_FOLDER.mkdir(parents=True, exist_ok=True)
        
        # 데이터셋 디렉토리 설정
        DATASET_PERSON_FOLDER.mkdir(parents=True, exist_ok=True)
        DATASET_CLOTH_FOLDER.mkdir(parents=True, exist_ok=True)
        
        model.create_model()

        print("Testing database connection...")
    except Exception as e:
        print(f"모델 로딩 오류: {str(e)}")
        print("Continuing without model...")

    try:
        connection = get_db_connection()
        connection.close()
        print("Database connection successful")
        yield
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        raise



# 현재 스크립트의 디렉토리를 기준으로 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
webserver2_dir = os.path.join(parent_dir, 'WebServer2')
static_dir = os.path.join(webserver2_dir, "static")

# 경로 정보 출력
print("\n=== 경로 정보 ===")
print(f"현재 스크립트 디렉토리: {current_dir}")
print(f"WebServer2 디렉토리: {webserver2_dir}")
print(f"Static 디렉토리: {static_dir}")
print(f"Static 디렉토리 존재 여부: {os.path.exists(static_dir)}")
if os.path.exists(static_dir):
    print("Static 디렉토리 내용:")
    for item in os.listdir(static_dir):
        print(f"  - {item}")
print("================\n")


# 템플릿과 정적 파일 설정
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# 결과 저장 디렉토리 설정
UPLOAD_FOLDER = Path(current_dir) / 'static/uploads'
RESULT_FOLDER = Path(current_dir) / 'static/result'
SELECTED_PERSON_FOLDER = Path(current_dir) / 'static/selectedimage/test/image'
SELECTED_CLOTH_FOLDER = Path(current_dir) / 'static/selectedimage/test/Cloth'

# 데이터셋 디렉토리 설정
DATASET_PERSON_FOLDER = Path(current_dir) / 'static/dataset/test/person'
DATASET_CLOTH_FOLDER = Path(current_dir) / 'static/dataset/test/cloth'

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
SELECTED_PERSON_FOLDER.mkdir(parents=True, exist_ok=True)
SELECTED_CLOTH_FOLDER.mkdir(parents=True, exist_ok=True)
DATASET_PERSON_FOLDER.mkdir(parents=True, exist_ok=True)
DATASET_CLOTH_FOLDER.mkdir(parents=True, exist_ok=True)


app = FastAPI(lifespan=lifespan)
# 정적 파일 마운트
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# MySQL 연결 설정
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="FITCLO1234",
            database="fitclo"
        )
        print("Database connection successful")
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        raise HTTPException(status_code=500, detail=f"데이터베이스 연결 오류: {str(e)}")

# 비밀번호 해시 함수
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# 회원가입 요청 모델
class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    name: str
    phone_number: str
    birth_date: date
    gender: str
    height_cm: int
    weight_kg: int
    address: str

# 로그인 요청 모델
class UserLogin(BaseModel):
    email: str
    password: str

# 토큰 응답 모델
class Token(BaseModel):
    access_token: str
    token_type: str

class ProfileUpdate(BaseModel):
    username: str
    email: str
    phone_number: str
    birth_date: date
    height_cm: int
    weight_kg: int
    address: str

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register")
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/fitting")
async def fitting(request: Request):
    return templates.TemplateResponse("fitting.html", {"request": request})

@app.post("/register")
async def register(user: UserRegister):
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # 이메일 중복 확인
        cursor.execute("SELECT * FROM users WHERE email = %s", (user.email,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="이미 등록된 이메일입니다.")
        
        # 비밀번호 해시화
        hashed_password = hash_password(user.password)
        
        # 사용자 정보 저장
        cursor.execute("""
            INSERT INTO users (username, email, password, name, phone_number, 
                             birth_date, gender, height_cm, weight_kg, address)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (user.username, user.email, hashed_password, user.name, 
              user.phone_number, user.birth_date, user.gender, user.height_cm, 
              user.weight_kg, user.address))
        
        connection.commit()
        return {"message": "회원가입이 완료되었습니다."}
    
    except Exception as e:
        connection.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        connection.close()

#def create_access_token(data: dict):
#    to_encode = data.copy()
#    expire = datetime.utcnow() + timedelta(days=7)
#    to_encode.update({"exp": expire})
#    encoded_jwt = jwt.encode(to_encode, "your-secret-key", algorithm="HS256")
#    return encoded_jwt

# JWT 설정
SECRET_KEY = "your-secret-key-fitclo"  # 실제 운영 환경에서는 보안성 높은 키 사용
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.post("/login")
async def login(user: UserLogin):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # 이메일로 사용자 찾기
        cursor.execute("SELECT * FROM users WHERE email = %s", (user.email,))
        db_user = cursor.fetchone()
        
        if not db_user:
            raise HTTPException(status_code=400, detail="등록되지 않은 이메일입니다.")
        
        # 비밀번호 확인
        hashed_password = hash_password(user.password)
        if db_user['password'] != hashed_password:
            raise HTTPException(status_code=400, detail="비밀번호가 일치하지 않습니다.")
        
        # JWT 토큰 생성
        access_token = create_access_token({"sub": user.email})
        return Token(access_token=access_token, token_type="bearer")
    
    finally:
        cursor.close()
        connection.close()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Not authenticated", # 사용자에게 표시되는 오류 메시지에 맞게 수정
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]) # 수정: SECRET_KEY 및 ALGORITHM 변수 사용
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except PyJWTError:
        raise credentials_exception
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        if user is None:
            raise credentials_exception
        return user
    finally:
        cursor.close()
        connection.close()

@app.get("/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    # 민감한 정보 제외
    user_info = {
        "username": current_user["username"],
        "email": current_user["email"],
        "name": current_user["name"],
        "phone_number": current_user["phone_number"],
        "birth_date": current_user["birth_date"],
        "gender": current_user["gender"],
        "height_cm": current_user["height_cm"],
        "weight_kg": current_user["weight_kg"],
        "address": current_user["address"]
    }
    return user_info

@app.get("/profile")
async def profile_page(request: Request, current_user: dict = Depends(get_current_user)):
    try:
        # current_user는 get_current_user 함수를 통해 이미 검증되고
        # 데이터베이스에서 가져온 사용자 정보입니다.
        # birth_date를 문자열로 변환 (템플릿에서 날짜 형식으로 표시하기 위함)
        if isinstance(current_user.get("birth_date"), date):
            current_user["birth_date"] = current_user["birth_date"].strftime("%Y-%m-%d")
            
        return templates.TemplateResponse("profile.html", {
            "request": request,
            "user": current_user
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_profile")
async def update_profile(profile: ProfileUpdate, request: Request):
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # 프로필 업데이트
        cursor.execute("""
            UPDATE users 
            SET username = %s, phone_number = %s, birth_date = %s,
                height_cm = %s, weight_kg = %s, address = %s
            WHERE email = %s
        """, (profile.username, profile.phone_number, profile.birth_date,
              profile.height_cm, profile.weight_kg, profile.address, profile.email))
        
        connection.commit()
        return {"message": "프로필이 업데이트되었습니다."}
    
    except Exception as e:
        connection.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        connection.close()

def find_config_path():
    """설정 파일의 경로를 찾습니다."""
    config_path = os.path.join(stableviton_dir, 'configs/VITONHD.yaml')
    if os.path.exists(config_path):
        return config_path
    return None

def find_model_path():
    """모델 파일의 경로를 찾습니다."""
    # 여러 가능한 경로 시도
    possible_paths = [
        os.path.join(stableviton_dir, 'ckpts/VITONHD_PBE_pose.ckpt'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

@app.post("/try-on")
async def try_on():
    try:
        # 선택된 이미지 찾기
        person_images = list(SELECTED_PERSON_FOLDER.glob("*"))
        cloth_images = list(SELECTED_CLOTH_FOLDER.glob("*"))

        if not person_images or not cloth_images:
            raise HTTPException(status_code=400, detail="선택된 이미지를 찾을 수 없습니다. 이미지를 다시 선택해주세요.")

        person_path = str(person_images[0])
        cloth_path = str(cloth_images[0])

        # test_pairs.txt 파일 생성
        pairs_file = SELECTED_PERSON_FOLDER.parent.parent / "test_pairs.txt"
        with open(pairs_file, "w") as f:
            person_filename = os.path.basename(person_path)
            cloth_filename = os.path.basename(cloth_path)
            f.write(f"{person_filename} {cloth_filename}\n")

        print("test_pairs.txt 파일 생성 완료")

        # 이미지 파일 복사 및 구조 확인
        test_dir = SELECTED_PERSON_FOLDER.parent / "test"
        test_dir.mkdir(exist_ok=True)

        print("이미지 복사 완료")

        # 모델 추론
        model.set_dataset()
        model.inference()

        # 결과 이미지 경로 생성
        person_name = os.path.splitext(os.path.basename(person_path))[0]
        cloth_name = os.path.splitext(os.path.basename(cloth_path))[0]
        result_filename = f"{person_name}_{cloth_name}.jpg"
        result_path = f"/static/result/{result_filename}"  # 클라이언트에서 접근 가능한 URL 경로

        # 결과 파일이 실제로 존재하는지 확인
        actual_result_path = os.path.join(current_dir, "static", "result", result_filename)
        if not os.path.exists(actual_result_path):
            raise HTTPException(status_code=500, detail="결과 이미지 생성에 실패했습니다.")

        return {"status": "success", "result_path": result_path}

    except Exception as e:
        print(f"가상 피팅 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/save-selected-images")
async def save_selected_images(
    person: Optional[UploadFile] = File(None),
    cloth: Optional[UploadFile] = File(None),
    person_path: Optional[str] = Form(None),
    cloth_path: Optional[str] = Form(None)
):
    try:
        response_data = {"status": "success"}

        if person or person_path:
            # 기존 선택된 인물 이미지 삭제
            for file in SELECTED_PERSON_FOLDER.glob("*"):
                file.unlink()

            if person:
                # 업로드된 인물 이미지 저장
                filename = person.filename
                person_file_path = SELECTED_PERSON_FOLDER / filename
                with open(person_file_path, "wb") as buffer:
                    shutil.copyfileobj(person.file, buffer)
                print(f"선택된 인물 이미지 저장됨: {person_file_path}")
                response_data["person_path"] = str(person_file_path.relative_to(Path(current_dir) / 'static'))
            elif person_path:
                # 예제 인물 이미지 복사
                source_path = Path(current_dir) / person_path
                filename = source_path.name
                person_file_path = SELECTED_PERSON_FOLDER / filename
                shutil.copy2(source_path, person_file_path)
                print(f"예제 인물 이미지 복사됨: {person_file_path}")
                response_data["person_path"] = str(person_file_path.relative_to(Path(current_dir) / 'static'))

        if cloth or cloth_path:
            # 기존 선택된 의상 이미지 삭제
            for file in SELECTED_CLOTH_FOLDER.glob("*"):
                file.unlink()

            if cloth:
                # 업로드된 의상 이미지 저장
                filename = cloth.filename
                cloth_file_path = SELECTED_CLOTH_FOLDER / filename
                with open(cloth_file_path, "wb") as buffer:
                    shutil.copyfileobj(cloth.file, buffer)
                print(f"선택된 의상 이미지 저장됨: {cloth_file_path}")
                response_data["cloth_path"] = str(cloth_file_path.relative_to(Path(current_dir) / 'static'))
            elif cloth_path:
                # 예제 의상 이미지 복사
                source_path = Path(current_dir) / cloth_path
                filename = source_path.name
                cloth_file_path = SELECTED_CLOTH_FOLDER / filename
                shutil.copy2(source_path, cloth_file_path)
                print(f"예제 의상 이미지 복사됨: {cloth_file_path}")
                response_data["cloth_path"] = str(cloth_file_path.relative_to(Path(current_dir) / 'static'))

        if not (person or person_path or cloth or cloth_path):
            raise HTTPException(status_code=400, detail="이미지가 선택되지 않았습니다.")
            
        return response_data
        
    except Exception as e:
        print(f"이미지 저장 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-sample-images")
async def get_sample_images(page: int = 1, per_page: int = 20):
    try:
        # 인물 이미지 목록 가져오기
        person_images = []
        if DATASET_PERSON_FOLDER.exists():
            all_person_images = sorted([f"/static/dataset/test/person/{f.name}"
                                     for f in DATASET_PERSON_FOLDER.glob('*.jpg')])
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            person_images = all_person_images[start_idx:end_idx]
            total_person_pages = (len(all_person_images) + per_page - 1) // per_page

        # 의상 이미지 목록 가져오기
        cloth_images = []
        if DATASET_CLOTH_FOLDER.exists():
            all_cloth_images = sorted([f"/static/dataset/test/cloth/{f.name}"
                                    for f in DATASET_CLOTH_FOLDER.glob('*.jpg')])
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            cloth_images = all_cloth_images[start_idx:end_idx]
            total_cloth_pages = (len(all_cloth_images) + per_page - 1) // per_page

        print("Found person images:", person_images)
        print("Found cloth images:", cloth_images)

        return {
            "person_images": person_images,
            "cloth_images": cloth_images,
            "total_person_pages": total_person_pages,
            "total_cloth_pages": total_cloth_pages,
            "current_page": page
        }
    except Exception as e:
        print(f"샘플 이미지 로딩 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/select-sample-image")
async def select_sample_image(request: Request):
    try:
        data = await request.json()
        image_path = data.get("image_path")
        image_type = data.get("type")

        if not image_path or not image_type:
            raise HTTPException(status_code=400, detail="이미지 경로와 타입이 필요합니다.")

        # 절대 경로로 변환
        source_path = Path(current_dir) / image_path.lstrip('/')
        
        # 대상 폴더 선택
        target_folder = SELECTED_PERSON_FOLDER if image_type == "person" else SELECTED_CLOTH_FOLDER
        
        # 기존 파일 삭제
        for file in target_folder.glob("*"):
            file.unlink()
        
        # 새 파일 복사
        target_path = target_folder / source_path.name
        shutil.copy2(source_path, target_path)
        
        return {"status": "success", "path": str(target_path)}
    except Exception as e:
        print(f"샘플 이미지 선택 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def prepare_dataset_structure(person_path, cloth_path):
    base_dir = "D:/kwj/fitclo/VSViton_serverTest/WebServer2/static/selectedimage/test"
    
    # 필요한 디렉토리 구조 생성
    os.makedirs(opj(base_dir, "image"), exist_ok=True)
    os.makedirs(opj(base_dir, "cloth"), exist_ok=True)
    os.makedirs(opj(base_dir, "agnostic-v3.2"), exist_ok=True)
    
    # 이미지 파일 복사
    person_filename = os.path.basename(person_path)
    cloth_filename = os.path.basename(cloth_path)
    
    # 인물 이미지 복사
    shutil.copy2(person_path, opj(base_dir, "image", person_filename))
    # 의상 이미지 복사
    shutil.copy2(cloth_path, opj(base_dir, "cloth", cloth_filename))
    # agnostic 이미지 복사 (인물 이미지와 동일한 이름 사용)
    shutil.copy2(person_path, opj(base_dir, "agnostic-v3.2", person_filename))
    
    return person_filename, cloth_filename

if __name__ == "__main__":
    import uvicorn
    #uvicorn.run("app:app", host="25.5.215.164", port=5001, reload=False)
    uvicorn.run("app:app", host="127.0.0.1", port=5001, reload=False)