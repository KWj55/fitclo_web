// 이미지 경로 설정
const personImagesPath = 'StableVITON1/dataset/test/image/';  // 인물 이미지 경로 (앞의 슬래시 제거)
const garmentImagesPath = 'StableVITON1/dataset/test/cloth/';  // 의상 이미지 경로 (앞의 슬래시 제거)

// DOM 요소
const personImageBox = document.getElementById('personImageBox');
const garmentImageBox = document.getElementById('garmentImageBox');
const personImageGrid = document.getElementById('personImageGrid');
const garmentImageGrid = document.getElementById('garmentImageGrid');
const selectedPersonImage = document.getElementById('selectedPersonImage');
const selectedGarmentImage = document.getElementById('selectedGarmentImage');
const runButton = document.getElementById('runButton');
const resultImage = document.getElementById('resultImage');

// 선택된 이미지 정보 저장
let selectedImages = {
    person: null,
    garment: null
};

// 실제 이미지 파일 목록
const personImages = [
    '13102_00.jpg', '13105_00.jpg', '13109_00.jpg', '13126_00.jpg', '13136_00.jpg',
    '13140_00.jpg', '13144_00.jpg', '13166_00.jpg', '13172_00.jpg', '13175_00.jpg',
    '13196_00.jpg', '13198_00.jpg', '13201_00.jpg', '13204_00.jpg', '13213_00.jpg'
];

const garmentImages = [
    '12723_00.jpg', '12724_00.jpg', '12736_00.jpg', '12741_00.jpg', '12748_00.jpg',
    '12749_00.jpg', '12750_00.jpg', '12755_00.jpg', '12781_00.jpg', '12789_00.jpg',
    '12801_00.jpg', '12807_00.jpg', '12810_00.jpg', '12813_00.jpg', '12818_00.jpg'
];

// 이미지 그리드 초기화 함수
async function initializeImageGrids() {
    try {
        loadImagesIntoGrid(personImages, personImageGrid, personImagesPath, 'person');
        loadImagesIntoGrid(garmentImages, garmentImageGrid, garmentImagesPath, 'garment');
    } catch (error) {
        console.error('이미지 로딩 중 오류 발생:', error);
    }
}

// 이미지를 그리드에 로드하는 함수
function loadImagesIntoGrid(images, grid, basePath, type) {
    grid.innerHTML = '';
    images.forEach(imageName => {
        const img = document.createElement('img');
        img.src = basePath + imageName;
        img.alt = imageName;
        img.classList.add('grid-image');
        img.addEventListener('click', () => selectImage(img, type, imageName));
        grid.appendChild(img);
    });
}

// 이미지 선택 함수
function selectImage(img, type, imageName) {
    // 이전 선택 제거
    const grid = type === 'person' ? personImageGrid : garmentImageGrid;
    grid.querySelectorAll('img').forEach(i => i.classList.remove('selected'));
    
    // 새로운 선택 추가
    img.classList.add('selected');
    selectedImages[type] = imageName;
    
    // 선택된 이미지 표시
    const targetImage = type === 'person' ? selectedPersonImage : selectedGarmentImage;
    targetImage.src = img.src;
    targetImage.style.display = 'block';
    
    // 실행 버튼 활성화 여부 체크
    checkRunButtonState();
}

// 실행 버튼 상태 체크
function checkRunButtonState() {
    runButton.disabled = !(selectedImages.person && selectedImages.garment);
}

// 실행 버튼 클릭 이벤트
runButton.addEventListener('click', async () => {
    
});

// 페이지 로드 시 초기화
initializeImageGrids(); 