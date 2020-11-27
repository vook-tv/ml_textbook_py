## [책따망 트러블 슈팅] 파이썬으로 배우는 머신러닝의 교과서
아래 내용들은 책에 수록된 실습 코드들의 트러블 슈팅입니다.

### CHAPTER 1. 머신러닝의 준비

* NVIDIA GPU가 없을 경우 책의 방법대로는 tensorflow 설치가 불가능하므로 다음과 같이 해결
    * **Step 1:** cudart64_101.dll을 로드할 수 없다는 오류 발생 시 tensorflow-cpu 설치([오류 상황 및 해결 영상](https://youtu.be/sVSRIdJmYek))
    * **Step 2:** (Windows Only) 환경 변수에 python38/script 경로 추가([환경변수 추가 영상](https://youtu.be/iDPwSHIDKXg))
    * **Note:** Anaconda Prompt와 같은 콘솔 모드에서 오류가 발생해도 Jupyter Notebook에서는 오류가 발생하지 않는 경우도 있습니다.
    * 전체 플레이리스트는 [여기서](https://www.youtube.com/playlist?list=PL3vETZ0d3GBz1p69OQn7dmO04yBIa0iXz)

* tensorflow를 import 하지 못하는 또 다른 케이스
    * **오류 상황:** _pywrap_tensorflow_internal 관련 DLL 오류(ImportError) 발생([오류 상황 영상: 16분 15초 부터 오류](https://youtu.be/Huejvbsa30M))
    * **해결 방법:** vc_redist (Microsoft Visual C++ Redistributable) 설치([설치 영상](https://youtu.be/5dkUATZj4no))
    * 설치 후 재부팅 필요(참조: https://needneo.tistory.com/47)
    * 전체 플레이리스트는 [여기서](https://www.youtube.com/playlist?list=PL3vETZ0d3GBwYfllUha6tVKo9U2Fsugy1)

### CHAPTER 4. 머신러닝에 필요한 수학의 기본
* 4.7.9(164p) 코드를 아래와 같이 수정해야 오류가 발생하지 않습니다. (np.reshape의 마지막 파라미터를 1에서 'F'로 변경)
* 참고한 출처는 [이곳](https://qiita.com/hiroshim021/items/535489b965b022d109c7) 입니다.
* 실습 동영상은 [여기를](https://youtu.be/7_0AyhhUWVg) 참고하세요. (등고선: 23분 50초, 3D: 26분)

#### # 등고선 표시
###### [수정 전]
    x = np.c_[np.reshape(xx0, nx * xn, 1), np.reshape(xx1, xn * xn, 1)]
##### [수정 후]
    x = np.c_[np.reshape(xx0, nx * xn, 'F'), np.reshape(xx1, xn * xn, 'F')]

#### # 3D 표시
###### [수정 전]
    x = np.c_[np.reshape(xx0, nx * xn, 1), np.reshape(xx1, xn * xn, 1)]
##### [수정 후]
    x = np.c_[np.reshape(xx0, nx * xn, 'F'), np.reshape(xx1, xn * xn, 'F')]


