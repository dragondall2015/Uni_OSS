import sys
import io
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
import random
from ytmusicapi import YTMusic

from model_apply import model, tokenizer, device, predict_emotion_with_probabilities

ytmusic = YTMusic("browser.json")

admin_chat_id = 

def get_mood_param_map():
    # 현재 모드 카테고리와 무드 리스트 가져오기
    mood_categories = ytmusic.get_mood_categories()
    moods = mood_categories.get("Moods & moments", [])
    # moods는 [{'title': "...", 'params': "..."}, ...]

    # title -> params 맵핑 딕셔너리 생성
    mood_map = {}
    for m in moods:
        mood_map[m['title']] = m['params']
    return mood_map

def get_print_emotion_probs(emotion_probs):
    sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
    lines = []
    for emotion, prob in sorted_emotions:
        percent = f"{prob*100:.0f}%"
        line = f"{emotion}: {percent}"
        lines.append(line)
    result = "\n".join(lines)
    return result

# 감정별 무드 리스트 정의
emotion_to_moods = {
    "분노": ["Energy Boosters"],
    "불안": ["Chill", "Commute", "Focus", "Sleep"],
    "놀람": ["Chill", "Commute", "Focus", "Sleep"],
    "슬픔": ["Sad"],
    "행복": ["Feel Good", "Party", "Romance", "Workout"]
}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print(update)
    welcome_message = (
        "안녕하세요! EmotionDJ입니다😊\n\n"
        "이 챗봇은 텍스트(일기, 문장, 감정 표현 등)를 분석하여, "
        "해당 감정에 어울리는 노래를 추천해드립니다😎\n\n"
        "사용 방법:\n"
        "1. 감정이 담긴 텍스트를 입력하세요.\n"
        "(예: 오늘은 기분이 너무 좋았어!)\n"
        "2. 제가 분석한 감정에 맞춰 어울리는 음악을 추천해드립니다."
    )
    await update.message.reply_text(
        welcome_message
    )

async def handle_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text  # 사용자가 보낸 텍스트
    print(f"사용자가 보낸 메시지: {user_text}")

    emotion, emotion_probs = predict_emotion_with_probabilities(user_text, model, tokenizer, device)
    await process_emotion_and_recommend(update, context, user_text, emotion, emotion_probs)

async def process_emotion_and_recommend(update: Update, context: ContextTypes.DEFAULT_TYPE, user_input: str, emotion: str, emotion_probs=None):
    try:
        mood_map = get_mood_param_map()
    except Exception as e:
        # ytmusic API 에러 등의 경우
        await send_error_message(update, context, str(e))
        return

    # 감정에 따른 무드 리스트 가져오기
    if emotion not in emotion_to_moods:
        await send_error_message(update, context, f"정의되지 않은 감정: {emotion}")
        return

    mood_list = emotion_to_moods[emotion]

    try:
        mood_params = [mood_map[m] for m in mood_list]
    except KeyError as e:
        # mood_map에 없는 무드가 요청됨 -> YTMusic API 결과 변경
        await send_error_message(update, context, f"{e} 무드를 찾을 수 없습니다.")
        return

    # user_text와 emotion을 context.user_data에 저장
    context.user_data['last_user_input'] = user_input
    context.user_data['last_emotion'] = emotion
    context.user_data['last_user_emotion_probs'] = emotion_probs

    print_emotion_probs = get_print_emotion_probs(emotion_probs)

    # 슬픔이면 무드가 Sad 하나이므로 랜덤 5곡
    if emotion == "슬픔":
        params = mood_params[0]
        await recommend_sad_or_upset_songs(update, context, params, emotion, print_emotion_probs)
    elif emotion == "분노":
        params = mood_params[0]
        await recommend_sad_or_upset_songs(update, context, params, emotion, print_emotion_probs)
    else:
        # 불안/놀람/행복 케이스
        # 해당 무드들의 params으로 각각 2곡 추천 후 합쳐서 보내기
        await recommend_songs(update, context, mood_params, emotion, print_emotion_probs)
    
async def send_error_message(update: Update, context: ContextTypes.DEFAULT_TYPE, error_message: str):
    # 에러 발생 시 사용자에게 알림
    await update.message.reply_text(
        "노래 추천에 이상이 생겼습니다.\n"
        "관리자에게 알림이 전송됐습니다.\n"
        "불편을 드려 죄송합니다🥹"
    )
    # 관리자에게 알림
    await context.bot.send_message(chat_id=admin_chat_id, text=f"[에러 발생]\n{error_message}")

async def recommend_songs(update: Update, context: ContextTypes.DEFAULT_TYPE, mood_params, emotion=None, print_emotion_probs=None):
    all_track_messages = []
    # mood_params 리스트를 순회하며 각 param을 처리
    for param in mood_params:
        # 각 param에 대해 최대 10개의 플레이리스트 가져오기
        playlists = ytmusic.get_mood_playlists(param)[:10]
        if not playlists:
            # 해당 param에 대해 플레이리스트가 없으면 건너뜀
            continue

        # 10개 중 하나의 플레이리스트를 랜덤으로 선택
        selected_playlist = random.choice(playlists)
        playlist_id = selected_playlist['playlistId']
        playlist_data = ytmusic.get_playlist(playlist_id)

        # 플레이리스트에서 최대 2곡 랜덤으로 선택
        track_count = min(len(playlist_data['tracks']), 2)
        if track_count == 0:
            # 트랙이 없는 경우 이 param은 스킵
            continue

        chosen_tracks = random.sample(playlist_data['tracks'], track_count)

        # 선택한 곡들의 정보를 all_track_messages에 추가            
        for track in chosen_tracks:
            track_number = len(all_track_messages) + 1 
            track_url = f"https://music.youtube.com/watch?v={track['videoId']}"
            track_info = (
                f"{track_number}. {track['title']}\n"
                f"아티스트: {track['artists'][0]['name']}\n"
                f"곡 들으러 가기: {track_url}"
                )
            all_track_messages.append(track_info)

    # 모든 param을 처리한 후, 곡이 하나도 없는 경우 처리
    if not all_track_messages:
        await update.message.reply_text("추천할 플레이리스트가 없습니다. 다시 입력해주세요🥹")
        return

    await send_recommendation_with_options(update, context, all_track_messages, emotion, print_emotion_probs)

async def recommend_sad_or_upset_songs(update: Update, context: ContextTypes.DEFAULT_TYPE, mood_param, emotion=None, print_emotion_probs=None):
    playlists = ytmusic.get_mood_playlists(mood_param)[:3]
    if not playlists:
        await update.message.reply_text("")
        return

    selected_playlist = random.choice(playlists)
    playlist_id = selected_playlist['playlistId']
    playlist_data = ytmusic.get_playlist(playlist_id)

    # 슬픔,분노의 경우 5곡 선택
    tracks = random.sample(playlist_data['tracks'], min(len(playlist_data['tracks']), 5))

    track_messages = []
    for track in tracks:
            track_number = len(track_messages) + 1 
            track_url = f"https://music.youtube.com/watch?v={track['videoId']}"
            track_info = (
                f"{track_number}. {track['title']}\n"
                f"아티스트: {track['artists'][0]['name']}\n"
                f"곡 들으러 가기: {track_url}"
                )
            track_messages.append(track_info)

    # 모든 param을 처리한 후, 곡이 하나도 없는 경우 처리
    if not track_messages:
        await update.message.reply_text("추천할 플레이리스트가 없습니다. 다시 입력해주세요🥹")
        return

    await send_recommendation_with_options(update, context, track_messages, emotion, print_emotion_probs)

async def send_recommendation_with_options(update: Update, context: ContextTypes.DEFAULT_TYPE, track_messages,emotion=None, print_emotion_probs=None):

    if emotion is not None:
        # recommendation_text = f"당신의 감정은 [{emotion}]입니다.\n\n⭐️추천 음악⭐️\n\n" + "\n\n".join(track_messages)
        recommendation_text = (
            "EmotionDJ가 당신의 글을 분석했어요!😊\n\n"
            f"{print_emotion_probs}\n\n"
            f"현재 당신의 감정 [{emotion}]을/를 바탕으로 추천된 음악입니다🎵\n\n"
            + "\n\n".join(track_messages)
        )
    else:
        recommendation_text = "⭐️추천 음악⭐️ \n\n" + "\n\n".join(track_messages)
    
    keyboard = [
        [InlineKeyboardButton("새로운 노래 추천 받기", callback_data="new_recommend")],
        [InlineKeyboardButton("다시 입력하기", callback_data="restart")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # 콜백쿼리가 있는지 확인
    if update.callback_query:
        # 콜백 쿼리로 들어온 경우
        query = update.callback_query
        await query.message.reply_text(recommendation_text, reply_markup=reply_markup)
    else:
        # 일반 메시지로 들어온 경우
        await update.message.reply_text(recommendation_text, reply_markup=reply_markup)

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "new_recommend":
        
        user_input = context.user_data.get("last_user_input", None)
        
        if user_input is None:
            await query.edit_message_text("이전에 입력한 문장이 없습니다. 다시 문장을 입력해주세요🥹")
            return
            
        emotion = context.user_data.get("last_emotion", None)
        emotion_probs = context.user_data.get("last_user_emotion_probs", None)
        await process_emotion_and_recommend(update, context, user_input, emotion, emotion_probs)
    elif data == "restart":
        chat_id = query.message.chat_id
        await context.bot.send_message(chat_id=chat_id, text="다시 문장을 입력해주세요.")

def main():
   
    TOKEN = ""
    
   
    application = Application.builder().token(TOKEN).build()

    
    application.add_handler(CommandHandler("start", start))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_message))
    application.add_handler(CallbackQueryHandler(handle_callback_query, pattern="^(new_recommend|restart)$"))

    application.run_polling()

if __name__ == "__main__":
    main()
