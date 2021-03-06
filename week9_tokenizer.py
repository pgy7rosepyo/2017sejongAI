from nltk.tokenize import sent_tokenize, \
        word_tokenize, WordPunctTokenizer

# Define input text
input_text = "US President Donald Trump addressed the South Korean parliament in SeoulWednesday, with a fiery speech that was full of criticism of the country's northern neighbor and particularly its leader Kim Jong Un. The more successful South Korea becomes,the more decisively you discredit the dark fantasy at the heart of the Kim regime" 

# Sentence tokenizer 
print("\nSentence tokenizer:")
print(sent_tokenize(input_text))

# Word tokenizer
print("\nWord tokenizer:")
print(word_tokenize(input_text))

# WordPunct tokenizer
print("\nWord punct tokenizer:")
print(WordPunctTokenizer().tokenize(input_text))
