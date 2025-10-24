import google.generativeai as genai

genai.configure()  # ✅ ya usa tu variable GOOGLE_API_KEY

model = genai.GenerativeModel("gemini-1.5-flash")

prompt = "Give me 3 creative vegetarian recipes using quinoa and avocado."

response = model.generate_content(prompt)
print(response.text)
