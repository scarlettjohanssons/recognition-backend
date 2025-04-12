from app.services.preprocessing import convert_mp3_to_wav

# Конвертация всех mp3 файлов в wav
convert_mp3_to_wav("app/data/ferrari", "app/data/ferrari_wav")
convert_mp3_to_wav("app/data/audi", "app/data/audi_wav")
convert_mp3_to_wav("app/data/other", "app/data/other_wav")
convert_mp3_to_wav("app/data/test", "app/data/test_wav")
convert_mp3_to_wav("app/data/noises", "app/data/noises_wav")

print("Преобразование завершено!")
