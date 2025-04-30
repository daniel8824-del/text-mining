// 결과를 표시하는 함수
function displayResults(data) {
    // 기존 결과 표시 로직
    // ... existing code ...

    // 다운로드 버튼 표시
    document.getElementById('download-csv').style.display = 'inline-block';
}

// CSV 다운로드 기능
document.getElementById('download-csv').addEventListener('click', function () {
    // 현재 분석 결과 데이터가 있는지 확인
    if (!currentData) {
        alert('다운로드할 분석 결과가 없습니다. 먼저 분석을 실행해주세요.');
        return;
    }

    // 서버로 데이터 전송하여 CSV 다운로드
    fetch('/download_csv', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(currentData)
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('CSV 다운로드 중 오류가 발생했습니다.');
            }
            return response.blob();
        })
        .then(blob => {
            // 파일 다운로드 처리
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = '텍스트_분석_결과.csv';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
        })
        .catch(error => {
            console.error('다운로드 오류:', error);
            alert('다운로드 중 오류가 발생했습니다: ' + error.message);
        });
});

// 전역 변수로 현재 분석 결과 데이터 저장
let currentData = null;

// 분석 실행 함수 수정
document.getElementById('analyze-button').addEventListener('click', function () {
    // ... existing code ...

    fetch('/analyze', {
        // ... existing code ...
    })
        .then(response => response.json())
        .then(data => {
            // 분석 결과 데이터를 전역 변수에 저장
            currentData = data;
            displayResults(data);
        })
        .catch(error => {
            // ... existing code ...
        });
}); 