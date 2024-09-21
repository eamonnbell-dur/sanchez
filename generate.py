"""
Requires ollama to be installed.
Usage: parallel -j 1 --bar "python generate.py --minutes {1} --gender {2} 
    --genre {3} --specialty {4} --output_file output/{1}_{2}_{3}_{4}__attempt_{5}.txt"
    ::: 5 10 60 ::: male female ::: fugue Lied ::: pianist guitarist ::: `seq 1 20`
"""
import argparse
import ollama
import tqdm
import torch

def generate_transcript(minutes, gender, genre, specialty, output_file):
    HISTORY_LENGTH = 4000  

    system_prompt = f"""
    You are a {gender} {specialty}. You are composing a {genre}. 

    You are participating in a cognitive science experiment and you are asked to "think-aloud" while you complete this task. 

    You are recording the thoughts that occur to you as you complete this task. You speak your thoughts out loud and they are recorded on an audio tape. 

    It will take approximately {minutes} minutes to complete the whole task. Timestamps should be in the format [mm:ss]. Don't provide any preface or header. Don't say "here is the transcript", or similar, just produce the transcript.

    The transcript should be consistent with the most recent thoughts in transcript so far:
    """

    history = ""
    transcript = ""
    summarise_calls = 0
    with open(output_file, "w") as file:
        with tqdm.tqdm(range(minutes), desc="Generating Transcript") as pbar:
            for minute in pbar:
                torch.cuda.empty_cache()

                seconds_start = minute * 60
                timestamp_start = f"{seconds_start // 3600:02}:{(seconds_start % 3600) // 60:02}:{seconds_start % 60:02}"

                seconds_end = ((minute + 1) * 60 - 1)
                timestamp_end = f"{seconds_end // 3600:02}:{(seconds_end % 3600) // 60:02}:{seconds_end % 60:02}"

                user_message = f"Give me the transcript of this audio tape from {timestamp_start} to {timestamp_end}."

                prompt = f"{system_prompt}\n\n{history}\n\n{user_message}"

                response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
                response_text = response['message']['content']
                
                history += f"{response_text}"
                file.write(f"{response_text}\n")
                
                if len(history) > HISTORY_LENGTH:
                    summarise_calls += 1
                    history = history[-HISTORY_LENGTH:]
                
                pbar.set_postfix({"call_length": (len(system_prompt) + len(history) + len(user_message)), 
                                "transcript_length": len(transcript), 
                                "summarise_calls": summarise_calls})

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate transcript from audio tape.")
    parser.add_argument("--minutes", type=int, default=5, help="Duration in minutes for the transcript generation.")
    parser.add_argument("--gender", type=str, default="female", help="Gender of the speaker.")
    parser.add_argument("--genre", type=str, default="fugue", help="Genre of the composition.")
    parser.add_argument("--specialty", type=str, default="pianist", help="Specialty of the speaker.")
    parser.add_argument("--output_file", type=str, default="transcript.txt", help="Output file to save the transcript.")
    
    args = parser.parse_args()
    generate_transcript(args.minutes, args.gender, args.genre, args.specialty, args.output_file)
