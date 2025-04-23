# Create a test script named test_masking.py
from utils import PIIMasker

# Initialize the masker
masker = PIIMasker()

# Test with some sample emails containing PII
test_emails = [
    "Hi, my name is John Smith and my email is john.smith@example.com. My phone number is 555-123-4567.",
    "Please charge my credit card 4111-1111-1111-1111 with expiry 12/25 and CVV 123.",
    "My Aadhar number is 1234 5678 9012 and I was born on 15/05/1985."
]

# Test each email
for i, email in enumerate(test_emails):
    print(f"\nTest Email {i+1}:")
    print("-" * 50)
    print(f"Original: {email}")
    
    masked_email, entities = masker.mask_text(email)
    
    print(f"\nMasked: {masked_email}")
    print("\nDetected Entities:")
    for entity in entities:
        print(f"  • {entity['classification']}: '{entity['entity']}' at position {entity['position']}")
    print("-" * 50)