# 크롤링 종류
1. 정적 웹 페이지 크롤링  
> 정적 웹 페이지란 서버(Web Server)에 <u>미리 저장된 파일</u>이 그대로 전달되는 웹 페이지를 말합니다. 즉, 특정 웹페이지의 url 주소만 주소창에 입력하면 웹 브라우저로 HTML 정보를 마음대로 가져올 수 있습니다. 
- 장점
  - 요청에 대한 파일만 전송하면 되기 때문에 서버간 통신이 거의 없고 속도가 빠름
  - 단순한 문서들로만 이루어져 있어서 어떤 호스팅서버에서도 동작 가능하므로 구축하는데 드는 비용이 적음
- 단점
  - 저장된 정보만 보여주기 때문에 서비스가 한정적
  - 추가, 수정, 삭제 등의 작업을 서버에서 직접 다운받아 편집 후 업로드로 수정해줘야 하기 때문에 관리가 어려움
- 사용 라이브러리: requests / BeautifulSoup
2. 동적 웹 페이지 크롤링
> 동적 웹 페이지란 입력, 클릭, 로그인 등 여러 이유로 [Ajax(비동기 통신)](https://sjparkk-dev1og.tistory.com/27) 형태로 서버(Was Server)와 데이터를 주고 받아 동적으로 제공하는 웹 페이지를 말합니다. 
- 장점
  - 다양한 정보를 조합하여 웹 페이지를 제공하기 때문에 서비스가 다양함
  - 추가, 수정, 삭제 등의 작업이 가능하기 때문에 관리가 편함
- 단점
  - 웹 페이지를 보여주기 위해서 여러번의 비동기 통신을 처리하기 때문에 상대적으로 속도가 느림
  - Web Server외에 Was Server가 추가로 필요함
- 사용 라이브러리: selenium, chromedriver


# 모듈들
## [pyautogui](https://codetorial.net/pyautogui/index.html)
- 파이썬 마우스/키보드 자동 조작 모듈 
- pip install pyautogui
- pip install pyperclip 

## [requests](https://www.daleseo.com/python-requests/)
- 파이썬 HTTP 통신에 사용되는 모듈
- pip install requests

## [beautifulsoup](https://wikidocs.net/85739)
- HTML, XML, JSON 등 파일의 구문을 분석하는 모듈
- pip install beautifulsoup4

## [selenium](https://gorokke.tistory.com/8)
- pip install selenium
### 브라우저별 드라이버 설치링크
- [Firefox](https://github.com/mozilla/geckodriver/releases)
- [Chrome](https://chromedriver.chromium.org/downloads)


# 참고문서 
- https://goodthings4me.tistory.com/491
- https://coding-kindergarten.tistory.com/category/%ED%8C%8C%EC%9D%B4%EC%8D%AC%20%ED%8C%A8%ED%82%A4%EC%A7%80/%EC%9B%B9%20%ED%81%AC%EB%A1%A4%EB%A7%81
- https://wikidocs.net/85383

