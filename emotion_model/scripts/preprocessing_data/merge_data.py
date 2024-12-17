
### 사용한 데이터섯 (AI 허브)

'''
## 감성대화 말뭉치
https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=86
columns 
0 : 인덱스 번호, 1 : 연령, 2 : 상황키워드, 
3 : 신체질환, 4 : 감정 대분류, 5 : 감정 소분류,
6 : 사람문장1, 7 : 시스템문장2, 8 : 사람문장2, 
9 : 시스템문장2, 10 : 사람문장 : 3, 11 : 시스템 문장 3

감정 대분류 : 기쁨 당황 분노 불안 상처 슬픔
감정 소분류 : ['노여워하는' '느긋' '걱정스러운' '당혹스러운' '당황' '마비된' '만족스러운' '배신당한' '버려진' '부끄러운' '분노'
 '불안' '비통한' '상처' '성가신' '스트레스 받는' '슬픔' '신뢰하는' '신이 난' '실망한' '악의적인' '안달하는'
 '안도' '억울한' '열등감' '염세적인' '외로운' '우울한' '고립된' '좌절한' '후회되는' '혐오스러운' '한심한'
 '자신하는' '기쁨' '툴툴대는' '남의 시선을 의식하는' '회의적인' '죄책감의' '혼란스러운' '초조한' '흥분' '충격 받은'
 '취약한' '편안한' '방어적인' '질투하는' '두려운' '눈물이 나는' '짜증내는' '조심스러운' '낙담한' '환멸을 느끼는'
 '희생된' '감사하는' '구역질 나는' '괴로워하는' '가난한, 불우한']
'''

'''
## 한국어 단발성 대화 데이터셋
https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=270
colums 
0 : Sentens,  1 : Emotion

감정분류 : 공포 놀람 분노 슬픔 중립 행복 혐오
'''

# 최종 맵핑 감정  : 행복 놀람 분노 슬픔 불안

import pandas as pd

## 감성대화 말뭉치
# 사람 문장 1, 2, 3 열과 감정 대분류 열만 추출 2개의 칼럼으로(emotion, text)
emotion_list = pd.read_excel('../../non_classified_data/emotion_list.xlsx')
emotion_list['감정_대분류'] = emotion_list['감정_대분류'].replace('공포', '불안')  # '공포'를 '불안' 으로 통일
emotion_list_two_columns = pd.DataFrame({
    'emotion': emotion_list['감정_대분류'],
    'text': emotion_list['사람문장1'].fillna('') + ' ' +
               emotion_list['사람문장2'].fillna('') + ' ' +
               emotion_list['사람문장3'].fillna('')
})

# 불필요한 공백 제거
emotion_list_two_columns['text'] = emotion_list_two_columns['text'].str.strip()
emotion_list_two_columns.to_csv('../../data_set/emotion_two_columns.csv', index=False, encoding='utf-8-sig')

# 사람 문장 1, 2, 3 열과 감정 소분류 열만 추출 2개의 칼럼으로(emotion, text)
sub_emotion_list_two_columns = pd.DataFrame({
    'emotion': emotion_list['감정_소분류'],
    'text': emotion_list['사람문장1'].fillna('') + ' ' +
               emotion_list['사람문장2'].fillna('') + ' ' +
               emotion_list['사람문장3'].fillna('')
})

# 사람 문장 1, 2, 3 열과 감정 소분류를 합치고 대분류 , 텍스트로 추출 2개의 칼럼으로(emotion, text(텍스트+감정소분류)
emotion_merge_subemotion_text = pd.DataFrame({
    'emotion': emotion_list['감정_대분류'],
    'text': emotion_list['감정_소분류'] + ' ' +
                emotion_list['사람문장1'].fillna('') + ' ' +
                emotion_list['사람문장2'].fillna('') + ' ' +
                emotion_list['사람문장3'].fillna('')
})
                
# 불필요한 공백 제거
emotion_list_two_columns['text'] = emotion_list_two_columns['text'].str.strip()
emotion_list_two_columns.to_csv('../../data_set/emotion_two_columns.csv', index=False, encoding='utf-8-sig')

sub_emotion_list_two_columns['text'] = sub_emotion_list_two_columns['text'].str.strip()
sub_emotion_list_two_columns.to_csv('../../data_set/sub_emotion_two_columns.csv', index=False, encoding='utf-8-sig')

emotion_merge_subemotion_text['text'] = emotion_merge_subemotion_text['text'].str.strip()
emotion_merge_subemotion_text.to_csv('../../data_set/emotion_merge_subemotion_text.csv', index=False, encoding='utf-8-sig')


## 한국어 단발성 대화 데이터셋
# Emotion Sentens 칼럼을 2개의 칼럼으로(emotion, text)
korea_list = pd.read_excel('../../non_classified_data/korea_list.xlsx')
korea_list_two_csv = pd.DataFrame({
    'emotion': korea_list['Emotion'],
    'text': korea_list['Sentence']
})

# 불필요한 공백 제거, "중립" emotion을 제외
korea_list_two_csv['text'] = korea_list_two_csv['text'].str.strip()
korea_list_two_csv = korea_list_two_csv[korea_list_two_csv['emotion'] != '중립']
korea_list_two_csv.to_csv('../data_set/korea_two_columns.csv', index=False, encoding='utf-8-sig')

## 최종 파일 통합 부분 추가
# 개별 CSV 파일 불러오기
emotion_df = pd.read_csv('../../data_set/emotion_two_columns.csv')
korea_df = pd.read_csv('../../data_set/korea_two_columns.csv')

# 두 파일을 하나로 병합
combined_df = pd.concat([emotion_df, korea_df], ignore_index=True)

# 최종 감정 매핑 정의
# 감정  : 행복 놀람 분노 슬픔 불안
emotion_mapping = {
    '기쁨': '행복',
    '행복': '행복',
    '당황': '놀람',
    '놀람': '놀람',
    '혐오': '분노',
    '분노': '분노',
    '슬픔': '슬픔',
    '상처': '슬픔',
    '불안': '불안',
    '공포': '불안'
}

# 최종 파일에서만 감정 매핑 적용
combined_df['emotion'] = combined_df['emotion'].replace(emotion_mapping)

# 최종 파일 저장
combined_df.to_csv('../../data_set/all_emotions_combined.csv', index=False, encoding='utf-8-sig')

print("emotion_two_columns.csv(감성대화 대분류 감정) 파일 생성")
print("sub_emotion_two_columns.csv(감성대화 소분류 감정) 파일 생성")
print("emotion_merge_subemotion_text.csv(감정대화 대분류 , 소분류 + 텍스트) 파일 생성")
print("korea_two_cloumns.csv(한국어 단발성 대화) 파일 생성")
print()
print("all_emotions_combined.csv(최종 병합 파일) 생성 완료")