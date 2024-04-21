import torch
from transformers import BertTokenizer, BertForSequenceClassification

def classify_topic_with_bert(text, model_name="bert-base-uncased", device="cpu"):
  """
  Classifies a given text using a pre-trained BERT model for topic classification.

  Args:
      text (str): The text to be classified.
      model_name (str, optional): The name of the pre-trained BERT model to use. Defaults to "bert-base-uncased".
      device (str, optional): The device to use for computation (CPU or GPU). Defaults to "cpu".

  Returns:
      tuple: A tuple containing the predicted topic label and the associated probability.
  """

  # Load pre-trained BERT tokenizer and model
  tokenizer = BertTokenizer.from_pretrained(model_name)
  model = BertForSequenceClassification.from_pretrained(model_name, num_labels=NUM_TOPICS)  # Replace NUM_TOPICS with actual number of topics

  # Preprocess text (tokenization, padding, etc.)
  encoded_text = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")

  # Move input to specified device
  encoded_text = encoded_text.to(device)

  # Perform classification
  with torch.no_grad():
      model.eval()
      outputs = model(**encoded_text)
      logits = outputs.logits.squeeze(0)  # Remove batch dimension if present

  # Get the predicted topic label (index) with the highest probability
  predicted_label = torch.argmax(logits).item()

  # Convert predicted label index to topic name (if available)
  if hasattr(model, "label_list"):
      predicted_topic = model.label_list[predicted_label]
  else:
      predicted_topic = predicted_label

  # Get the probability associated with the predicted topic
  probability = torch.softmax(logits, dim=0)[predicted_label].item()

  return predicted_topic, probability

# Example usage (assuming you have a trained BERT model)
text = "The Federal Reserve announced a surprise interest rate cut today."
predicted_topic, probability = classify_topic_with_bert(text)

print(f"Predicted topic: {predicted_topic}")
print(f"Probability: {probability:.4f}")