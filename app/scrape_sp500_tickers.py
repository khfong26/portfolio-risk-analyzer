import pandas as pd
import yfinance as yf
import time


def scrape_sp500_valid_tickers():
    # Scrape the S&P 500 tickers from Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        df = pd.read_html(url)[0]
    except Exception as e:
        print("Failed to fetch S&P 500 tickers:", e)
        return

    tickers = df['Symbol'].tolist()

    # Clean up tickers (some might have dots or special chars like BRK.B)
    tickers = [t.replace('.', '-') for t in tickers]  # e.g., BRK.B → BRK-B

    # Validate each ticker using yfinance
    valid_tickers = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="1d")
            if not hist.empty:
                valid_tickers.append(t)
                print(f"✅ {t}")
            else:
                print(f"❌ {t} (no data)")
        except Exception as e:
            print(f"❌ {t} (error: {e})")
        time.sleep(0.3)  # polite delay to avoid being rate-limited :)

    # Save to a txt file
    with open("valid_sp500_tickers.txt", "w") as f:
        for t in valid_tickers:
            f.write(t + "\n")

    print(f"\nSaved {len(valid_tickers)} valid tickers to 'valid_sp500_tickers.txt'.")

if __name__ == "__main__":
    scrape_sp500_valid_tickers()
