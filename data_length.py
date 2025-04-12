import os

ferrari_count = len(os.listdir("app/data/mel_spectrograms/ferrari"))
audi_count = len(os.listdir("app/data/mel_spectrograms/audi"))
unknown_count = len(os.listdir("app/data/mel_spectrograms/unknown"))

print(f"Ferrari: {ferrari_count}, Audi: {audi_count}, Unknown: {unknown_count}")
