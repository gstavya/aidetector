from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model = AutoModelForSequenceClassification.from_pretrained("./model_output/checkpoint-9000")
model.eval()

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = F.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()

    return {
        "prediction": pred_class,
        "probabilities": probs.tolist()[0]
    }

result = classify_text("As the play progresses, Macbeth’s descent into madness becomes more pronounced. Once a loyal and valiant soldier, he transforms into a tyrant willing to kill anyone who threatens his hold on power, including his close friend Banquo and the innocent family of Macduff. His moral compass deteriorates rapidly, and he begins to rely more heavily on the witches’ cryptic prophecies, which give him a false sense of invincibility. Meanwhile, Lady Macbeth, who once seemed the more ruthless of the two, is eventually overcome with guilt and descends into mental instability, famously sleepwalking and obsessively trying to wash imagined blood from her hands. Her eventual suicide marks a turning point for Macbeth, who becomes numb to life’s meaning and resigns himself to his fate. In the end, Macbeth is slain by Macduff, a symbol of justice and retribution, and order is restored to Scotland with Malcolm’s ascension to the throne. Through Macbeth’s rise and fall, Shakespeare paints a grim portrait of how ambition, when unrestrained by ethical boundaries, can lead to one's self-destruction.")
print("Predicted class:", result["prediction"])
print("Probabilities:", result["probabilities"])
