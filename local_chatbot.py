import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use DialoGPT-small for lightning-fast CPU performance!
MODEL_NAME = "microsoft/DialoGPT-small"

print("🤖 Loading lightning-fast local chat model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("\n🚀 Fast Local Chatbot Ready! Type 'exit' to quit.\n" + "-"*50)

chat_history_ids = None

while True:
    try:
        user_input = input("You: ")
        if not user_input.strip():
            continue
            
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\n🤖 Bot: See ya! Keep building! 🚀")
            break
        
        # Encode user input
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        
        # Append to history, but TRUNCATE if it gets too long to prevent lag
        if chat_history_ids is not None:
            # Keep only the last 200 tokens of history so it stays fast
            if chat_history_ids.shape[-1] > 200:
                chat_history_ids = chat_history_ids[:, -200:]
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids
        
        # Generate response quickly
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=400,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            temperature=0.7,
            do_sample=True
        )
        
        # Extract only the newly generated bot reply
        response_ids = chat_history_ids[:, bot_input_ids.shape[-1]:]
        bot_reply = tokenizer.decode(response_ids[:,:], skip_special_tokens=True)
        
        # Clean up any weird Reddit brackets or quotes if they sneak in
        bot_reply = bot_reply.strip().strip('[]"')
        
        print(f"🤖 AI: {bot_reply}\n")
        
    except (KeyboardInterrupt, EOFError):
        print("\n🤖 Bot: Session closed!")
        break