from openai import OpenAI

def test_openai_key():
    client = OpenAI()
    response = client.responses.create(
        model = "gpt-4.1",
        input = "Give me a joke about cBioPortal and cancer genomics. Only give me the joke, without any additional text before or after it."
    )

    print(response.output_text)

if __name__ == "__main__":
    test_openai_key()