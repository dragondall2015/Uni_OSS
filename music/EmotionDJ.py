from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from ytmusicapi import YTMusic
import random

ytmusic = YTMusic("browser.json")


mood_categories = ytmusic.get_mood_categories()
moods = mood_categories.get("Moods & moments", [])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("음악 검색", callback_data="search_music")],
        [InlineKeyboardButton("도움말", callback_data="help")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "안녕하세요! EmotionDJ입니다. 무엇을 도와드릴까요?",
        reply_markup=reply_markup
    )


async def menu_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    if query.data == "search_music":
        
        await show_mood_categories(query)
    elif query.data == "help":
        
        await query.edit_message_text(
            "EmotionDJ는 기분에 맞는 음악을 추천해드립니다.\n\n"
            "- '음악 검색'을 눌러 기분을 선택하세요.\n"
            "- 기분에 따라 추천되는 곡들을 확인하세요!\n\n"
            "다시 시작하려면 /start를 입력하세요."
        )


async def show_mood_categories(query) -> None:
    keyboard = [
        [InlineKeyboardButton(mood['title'], callback_data=mood['params'])]
        for mood in moods
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text("기분을 선택하세요:", reply_markup=reply_markup)


async def recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    mood_param = query.data

    # 플레이리스트 검색
    playlists = ytmusic.get_mood_playlists(mood_param)[:3]
    if not playlists:
        await query.edit_message_text("추천할 플레이리스트가 없습니다. 다른 기분을 선택해 보세요.")
        return

    # 랜덤 플레이리스트 선택
    selected_playlist = random.choice(playlists)
    playlist_id = selected_playlist['playlistId']
    playlist_data = ytmusic.get_playlist(playlist_id)

    # 랜덤으로 5곡 선택
    tracks = random.sample(playlist_data['tracks'], min(len(playlist_data['tracks']), 5))

   
    track_messages = []
    for track in tracks:
        track_url = f"https://music.youtube.com/watch?v={track['videoId']}"
        track_info = f"{track['title']} by {track['artists'][0]['name']}\n{track_url}"
        track_messages.append(track_info)

  
    await query.edit_message_text("추천된 노래:\n\n" + "\n\n".join(track_messages))


def main():
   
    TOKEN = ""
    
   
    application = Application.builder().token(TOKEN).build()

    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(menu_selection, pattern="^(search_music|help)$"))
    application.add_handler(CallbackQueryHandler(recommend_music, pattern="^(?!search_music|help).*$"))

   
    application.run_polling()

if __name__ == "__main__":
    main()
