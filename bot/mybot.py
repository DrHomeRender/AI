import os
import discord
from discord.ext import commands

# Intents 생성 및 필요한 권한 활성화
intents = discord.Intents.default()  # 기본 intents 활성화
intents.messages = True  # 메시지 이벤트 받기
intents.message_content = True  # 메시지 내용 읽기 권한 활성화
intents.guilds = True    # 서버 관련 이벤트 받기
intents.voice_states = True  # 음성 채널 상태 변경 이벤트 받기

# 음악 파일이 저장된 폴더 경로 설정
MUSIC_FOLDER = "/mnt/c/Music/"  # 수정된 경로
  # 여기에 음악 파일이 저장된 경로를 설정

# 봇 생성 시 Intents 전달
bot = commands.Bot(command_prefix='*', intents=intents)

@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')

@bot.command(name='join', help='Joins a voice channel')
async def join(ctx):
    if ctx.author.voice:
        channel = ctx.message.author.voice.channel
        await channel.connect()
    else:
        await ctx.send("You are not connected to a voice channel.")

@bot.command(name='leave', help='Leaves a voice channel')
async def leave(ctx):
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
    else:
        await ctx.send("I am not in a voice channel.")

@bot.command(name='play', help='Plays a song')
async def play(ctx, *, filename):
    if ctx.voice_client:
        # 파일 경로 생성
        filepath = os.path.join(MUSIC_FOLDER, filename)

        # 파일 존재 여부 확인
        if not os.path.exists(filepath):
            await ctx.send(f"The file `{filename}` does not exist in the music folder.")
            return

        # 현재 재생 중인 오디오 정지
        ctx.voice_client.stop()

        # FFmpeg로 파일 재생
        source = discord.FFmpegPCMAudio(filepath)
        ctx.voice_client.play(source, after=lambda e: print(f'Player error: {e}') if e else None)
        await ctx.send(f"Now playing: {filename}")
    else:
        await ctx.send("I am not connected to a voice channel. Use `*join` first.")

@bot.command(name='list', help='Lists all available songs in the music folder')
async def list_songs(ctx):
    # MUSIC_FOLDER에서 파일 목록 가져오기
    files = os.listdir(MUSIC_FOLDER)
    music_files = [f for f in files if f.endswith(('.mp3', '.wav'))]  # 지원하는 형식 필터링
    if music_files:
        await ctx.send("Available songs:\n" + "\n".join(music_files))
    else:
        await ctx.send("No songs found in the music folder.")



