from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="""
You are a senior financial market analyst.

Stock Symbol: {ticker}
Current Price: ${price_now}

Predicted Prices:
3 Month: ${p3}
6 Month: ${p6}
12 Month: ${p12}

Provide a professional investment analysis including:

1. Market trend interpretation
2. Short term outlook
3. Medium term outlook
4. Long term outlook
5. Potential risks
6. Investment recommendation (Buy / Hold / Sell)
""",
    input_variables=["ticker", "price_now", "p3", "p6", "p12"]
)

prompt.save("prompt_15.json")