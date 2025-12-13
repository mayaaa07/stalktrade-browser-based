import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# ----------------------------
# 1. Training Sentences (intents)
# ----------------------------
training_sentences = [
    # --- Add to watchlist ---
    "Add Apple to my watchlist",
    "Please add Tesla stock",
    "I want to add Microsoft",
    "Put Google in my watchlist",
    "Add Amazon to my list",
    "Track Apple stock for me",
    "Include Tesla in watchlist",
    "Save Microsoft stock",
    "Add Google to stocks",
    "Add Amazon stock",
    "Add Netflix to my watchlist",
    "Remove Reliance from my watchlist",
    "Buy Tata stocks",

    # --- Remove from watchlist ---
    "Remove Apple from watchlist",
    "Delete Tesla from my list",
    "Take Microsoft off my watchlist",
    "Remove Google stock",
    "Remove Amazon from watchlist",
    "Forget Apple stock",
    "Drop Tesla",
    "Clear Microsoft",
    "Delete Google",
    "Remove Amazon",

    # --- Check watchlist ---
    "Show my watchlist",
    "What stocks are in my watchlist?",
    "List my watchlist",
    "Check my watchlist",
    "Tell me my stocks",
    "What am I tracking?",
    "Watchlist please",
    "Which stocks are in my list?",
    "Display my watchlist",
    "Do I have Apple in my watchlist?",

    # --- Check portfolio ---
    "Show my portfolio",
    "What is in my portfolio?",
    "Check portfolio",
    "List my portfolio",
    "Tell me my holdings",
    "Display portfolio",
    "Portfolio details",
    "Open my portfolio",
    "What are my investments?",
    "Portfolio please",
]

# Labels (same length as training_sentences)
training_labels = [
    # Add
    "add_watchlist", "add_watchlist", "add_watchlist", "add_watchlist", "add_watchlist",
    "add_watchlist", "add_watchlist", "add_watchlist", "add_watchlist", "add_watchlist", "add_watchlist", "add_watchlist", "add_watchlist",

    # Remove
    "remove_watchlist", "remove_watchlist", "remove_watchlist", "remove_watchlist", "remove_watchlist",
    "remove_watchlist", "remove_watchlist", "remove_watchlist", "remove_watchlist", "remove_watchlist",

    # Check watchlist
    "check_watchlist", "check_watchlist", "check_watchlist", "check_watchlist", "check_watchlist",
    "check_watchlist", "check_watchlist", "check_watchlist", "check_watchlist", "check_watchlist",

    # Check portfolio
    "check_portfolio", "check_portfolio", "check_portfolio", "check_portfolio", "check_portfolio",
    "check_portfolio", "check_portfolio", "check_portfolio", "check_portfolio", "check_portfolio",
]


# ----------------------------
# 2. Q&A Knowledge Base
# ----------------------------
qa_data = [
    ("Tell me about Apple", 
     "Apple is one of the world’s largest technology companies:\n"
     "• Designs iPhone, Mac, iPad, Watch\n"
     "• Expanding services business\n"
     "• Strong balance sheet, safe-haven stock\n"
     "• Growth drivers: AI, AR/VR, wearables"),
    
    ("What is AAPL?", 
     "Apple Inc. (AAPL) is a trillion-dollar tech company:\n"
     "• Flagship products: iPhone, Mac, Services\n"
     "• Strong ecosystem integration\n"
     "• Popular with long-term investors"),
    
    ("Should I buy Apple stock?", 
     "Apple has strong fundamentals:\n"
     "• Consistent revenue growth\n"
     "• Huge brand loyalty\n"
     "• Safe for long-term, but depends on risk tolerance"),
]

# ----------------------------
# 3. Train Intent Model
# ----------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)

model = LogisticRegression(max_iter=1000)
model.fit(X, training_labels)

# ----------------------------
# 4. Save Models
# ----------------------------
with open("nlp_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("qa_data.pkl", "wb") as f:
    pickle.dump(qa_data, f)

print("✅ NLP intent model and QA data saved successfully!")
