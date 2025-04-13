from typing import List, Tuple

import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-960h",
            lm_model_path="/content/3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0
        ):
        """
        Initialization of Wav2Vec2Decoder class

        Args:
            model_name (str): Pretrained Wav2Vec2 model from transformers
            lm_model_path (str): Path to the KenLM n-gram model (for LM rescoring)
            beam_width (int): Number of hypotheses to keep in beam search
            alpha (float): LM weight for shallow fusion and rescoring
            beta (float): Word bonus for shallow fusion
        """
        # once logits are available, no other interactions with the model are allowed
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        # you can interact with these parameters
        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None
    def greedy_decode(self, logits: torch.Tensor) -> str:
        # Вычисляем лог-вероятности и предсказанные токены
        pred_ids = torch.argmax(torch.log_softmax(logits, dim=-1), dim=-1).tolist()
        
        # Удаляем повторы и пустые токены с помощью генератора
        decoded = [
            token_id for i, token_id in enumerate(pred_ids)
            if token_id != self.blank_token_id and (i == 0 or token_id != pred_ids[i - 1])
        ]
        
        # Преобразуем токены в строку
        transcript = ''.join(self.vocab[token_id] for token_id in decoded)
        return transcript.replace(self.word_delimiter, ' ')

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        import heapq
        # Apply log softmax to get log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        T, V = log_probs.shape
          
        # Initialize beam
        # (neg_score, prefix, prev_token)
        beam = [(0.0, [], self.blank_token_id)]
        
        # Process each time step
        for t in range(T):
            new_beam = []
            
            # For each hypothesis in the beam
            for score, prefix, prev_token in beam:
                # Option 1: Add blank token (keep same prefix)
                blank_score = score - log_probs[t, self.blank_token_id].item()
                new_beam.append((blank_score, prefix, self.blank_token_id))
                
                # For each token in vocabulary
                for v in range(V):
                    if v == self.blank_token_id:
                        continue  # Already handled blank
                    
                    new_score = score - log_probs[t, v].item()
                    
                    # Option 2: Token is different from previous
                    if v != prev_token:
                        new_beam.append((new_score, prefix + [v], v))
                    # Option 3: Token is same as previous (don't add to prefix)
                    else:
                        new_beam.append((new_score, prefix, v))
            
            # Keep only top beam_width hypotheses
            beam = heapq.nsmallest(self.beam_width, new_beam, key=lambda x: x[0])
        
        # Process final beam to get complete hypotheses
        # Финальная фильтрация и сортировка
        final_beam = [
            ([token for i, token in enumerate(prefix) if i == 0 or token != prefix[i - 1]], -score)
            for score, prefix, _ in beam
        ]
        beams = sorted(final_beam, key=lambda x: x[1], reverse=True)
        
        if return_beams:
            return beams
        best_hypothesis = ''.join(self.vocab[token_id] for token_id in beams[0][0])
        return best_hypothesis.replace(self.word_delimiter, ' ')

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        import heapq
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")
        log_probs = torch.log_softmax(logits, dim=-1)
        T, V = log_probs.shape
        beam = [(0.0, 0.0, [], self.blank_token_id, "")]  # (total_score, am_score, prefix, prev_token, prefix_str)
        
        for t in range(T):
            new_beam = []
            for total_score, am_score, prefix, prev_token, prefix_str in beam:
                # Обработка пустого токена
                blank_am_score = am_score - log_probs[t, self.blank_token_id].item()
                new_beam.append((blank_am_score, blank_am_score, prefix, self.blank_token_id, prefix_str))
                
                # Векторизация обработки всех остальных токенов
                valid_tokens = torch.arange(V)[log_probs[t] > -float('inf')]
                scores = am_score - log_probs[t, valid_tokens].cpu().numpy()
                for v, new_am_score in zip(valid_tokens, scores):
                    v = v.item()
                    char = self.vocab[v]
                    if v == self.blank_token_id:
                        continue
                    if v != prev_token:
                        new_prefix = prefix + [v]
                        new_prefix_str = prefix_str + char
                        lm_score = 0.0
                        if char == self.word_delimiter or char == ' ':
                            # Разделяем строку по разделителям
                            words = new_prefix_str.replace(self.word_delimiter, ' ').split()
                            if words:  # Проверяем, что список не пуст
                                last_word = words[-1]
                                lm_score = self.lm_model.score(last_word, bos=True, eos=False) + self.beta
                            else:
                                lm_score = 0.0  # Если слов нет, LM-оценка равна 0
                        else:
                            lm_score = 0.0
                        new_total_score = new_am_score - self.alpha * lm_score
                    
                        new_beam.append((new_total_score, new_am_score, new_prefix, v, new_prefix_str))

                    else:
                        new_beam.append((new_am_score, new_am_score, prefix, v, prefix_str))           
            # Выбор лучших гипотез
            beam = heapq.nsmallest(self.beam_width, new_beam, key=lambda x: x[0])
        
        # Финальная фильтрация и выбор лучшей гипотезы
        best_hypothesis = None
        best_score = float('-inf')
        for total_score, am_score, prefix, _, _ in beam:
            filtered_prefix = [token for i, token in enumerate(prefix) if i == 0 or token != prefix[i - 1]]
            transcript = ''.join(self.vocab[token_id] for token_id in filtered_prefix)
            transcript = transcript.replace(self.word_delimiter, ' ')
            lm_score = self.lm_model.score(transcript)
            final_score = -total_score + self.alpha * lm_score
            if final_score > best_score:
                best_score = final_score
                best_hypothesis = transcript
        return best_hypothesis

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """Second-pass LM rescoring implementation"""
        rescored = []
        for hyp, score in beams:
            text = "".join([self.vocab[t] for t in hyp]).replace(self.word_delimiter, " ")
            words = text.split()
            lm_score = self.lm_model.score(" ".join(words), bos=True, eos=True)
            total_score = score + self.alpha * lm_score + self.beta * len(words)
            rescored.append((text, total_score))

        return max(rescored, key=lambda x: x[1])[0].strip()
    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Decode input audio file using the specified method

        Args:
            audio_input (torch.Tensor): Audio tensor
            method (str): Decoding method ("greedy", "beam", "beam_lm", "beam_lm_rescore"),
                where "greedy" is a greedy decoding,
                      "beam" is beam search without LM,
                      "beam_lm" is beam search with LM shallow fusion, and
                      "beam_lm_rescore" is a beam search with second pass LM rescoring

        Returns:
            str: Decoded transcription
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError("Invalid decoding method. Choose one of 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'.")


def test(decoder, audio_path, true_transcription):

    import Levenshtein

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Audio sample rate must be 16kHz"

    print("=" * 60)
    print("Target transcription")
    print(true_transcription)

    # Print all decoding methods results
    for d_strategy in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        print("-" * 60)
        print(f"{d_strategy} decoding")
        transcript = decoder.decode(audio_input, method=d_strategy)
        print(f"{transcript}")
        print(f"Character-level Levenshtein distance: {Levenshtein.distance(true_transcription, transcript.strip())}")

if __name__ == "__main__":
    
    test_samples = [
        ("/content/examples/sample1.wav", "IF YOU ARE GENEROUS HERE IS A FITTING OPPORTUNITY FOR THE EXERCISE OF YOUR MAGNANIMITY IF YOU ARE PROUD HERE AM I YOUR RIVAL READY TO ACKNOWLEDGE MYSELF YOUR DEBTOR FOR AN ACT OF THE MOST NOBLE FORBEARANCE"),
        ("/content/examples/sample2.wav", "AND IF ANY OF THE OTHER COPS HAD PRIVATE RACKETS OF THEIR OWN IZZY WAS UNDOUBTEDLY THE MAN TO FIND IT OUT AND USE THE INFORMATION WITH A BEAT SUCH AS THAT EVEN GOING HALVES AND WITH ALL THE GRAFT TO THE UPPER BRACKETS HE'D STILL BE ABLE TO MAKE HIS PILE IN A MATTER OF MONTHS"),
        ("/content/examples/sample3.wav", "GUESS A MAN GETS USED TO ANYTHING HELL MAYBE I CAN HIRE SOME BUMS TO SIT AROUND AND WHOOP IT UP WHEN THE SHIPS COME IN AND BILL THIS AS A REAL OLD MARTIAN DEN OF SIN"),
        ("/content/examples/sample4.wav", "IT WAS A TUNE THEY HAD ALL HEARD HUNDREDS OF TIMES SO THERE WAS NO DIFFICULTY IN TURNING OUT A PASSABLE IMITATION OF IT TO THE IMPROVISED STRAINS OF I DIDN'T WANT TO DO IT THE PRISONER STRODE FORTH TO FREEDOM"),
        ("/content/examples/sample5.wav", "MARGUERITE TIRED OUT WITH THIS LONG CONFESSION THREW HERSELF BACK ON THE SOFA AND TO STIFLE A SLIGHT COUGH PUT UP HER HANDKERCHIEF TO HER LIPS AND FROM THAT TO HER EYES"),
        ("/content/examples/sample6.wav", "AT THIS TIME ALL PARTICIPANTS ARE IN A LISTEN ONLY MODE"),
        ("/content/examples/sample7.wav", "THE INCREASE WAS MAINLY ATTRIBUTABLE TO THE NET INCREASE IN THE AVERAGE SIZE OF OUR FLEETS"),
        ("/content/examples/sample8.wav", "OPERATING SURPLUS IS A NON CAP FINANCIAL MEASURE WHICH IS DEFINED AS FULLY IN OUR PRESS RELEASE"),
    ]

    decoder = Wav2Vec2Decoder()

    _ = [test(decoder, audio_path, target) for audio_path, target in test_samples]