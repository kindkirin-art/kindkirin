# 저장소 안내

## 일기 웹페이지 (GitHub Pages)

GitHub Pages는 `claude/diary-app-p9cjwg` 브랜치의 루트(`/`)를 서비스한다.
일기 웹페이지의 변경은 이 브랜치에 들어가야 웹주소에 반영된다.

| 페이지 | 파일 | 주소 |
|---|---|---|
| 두통일기 | `headache-diary/index.html` | https://kindkirin-art.github.io/kindkirin/headache-diary/ |
| 추세추종 일기 | `trend-diary/index.html` | https://kindkirin-art.github.io/kindkirin/trend-diary/ |

- 일기 페이지를 고쳐 달라는 요청을 받으면, 세션에 배정된 브랜치만으로는 웹주소에 반영되지 않는다는 점을
  사용자에게 알리고 `claude/diary-app-p9cjwg`에 반영할지 확인한다.
- 옛 브랜치 이름 `claude/headache-diary-app-p9cjwg`는 2026-11-11에 지금 이름으로 바뀌었다.
- 이 브랜치의 이름을 바꾸거나 삭제하면 Pages 설정을 다시 지정하기 전까지 사이트가 끊긴다.
- 새 페이지는 `<이름>/index.html` 폴더로 추가하면 `https://kindkirin-art.github.io/kindkirin/<이름>/`로 열린다.
- 각 페이지는 파일 하나로 동작하고, 기록은 사용자의 브라우저(localStorage)에만 저장된다.
- 푸시 후 반영까지 1~4분 걸린다.
