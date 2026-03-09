import os
os.environ["COMPLUS_Version"] = "v4.0.30319"

import json
import nsepython as nse
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException
from typing import Annotated
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import base64
import os
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta, date
import csv
from util.dbsetup import load_db
from util.auth import *
from util.model import get_stock_predictions
import asyncio
import logging

cur, conn = load_db()
nse_symbols = nse.nse_eq_symbols()

app = FastAPI(title="Finstox API")

async def run_nse_in_executor(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/logos", StaticFiles(directory="logos"), name="logos")
app.mount("/predictions", StaticFiles(directory="predictions"), name="predictions")

@app.get("/")
async def root():
    return {"message": "Internal API for Finstox"}

@app.get("/isMarketOpen")
async def isMarketOpen():
    status_result = await run_nse_in_executor(nse.nse_marketStatus)
    return status_result["marketState"][0]["marketStatus"]!="Closed";

# Search endpoint
@app.get("/getSearchSuggestions")
async def getSearchSuggestions(query: str):
    suggestions = []
    c=0
    q_lower = query.lower()
    for symbol in nse_symbols:
        if c == 3: break
        symbol_lower = symbol.lower()
        if symbol_lower.startswith(q_lower) or q_lower in symbol_lower:
            c+=1
            stock = await run_nse_in_executor(nse.nse_eq, symbol)
            if stock and 'info' in stock and 'priceInfo' in stock:
                suggestions.append({
                    "symbol": symbol,
                    "name" : stock["info"]["companyName"],
                    "ltp" : stock["priceInfo"]["lastPrice"]
                    })
            else:
                 print(f"Warning: Could not fetch data for {symbol} in getSearchSuggestions")
    return {"suggestions": suggestions}
    

@app.get("/getSearchSuggestionsFull")
async def getSearchSuggestionsFull(query: str):
    if len(query) < 3:
        return {"detail": "Query too short"}, 400
    suggestions = []
    q_lower = query.lower()
    match_count = 0
    for symbol in nse_symbols:
        if match_count >= 20: break
        symbol_lower = symbol.lower()
        if symbol_lower.startswith(q_lower) or q_lower in symbol_lower:
            match_count += 1
            stock = await run_nse_in_executor(nse.nse_eq, symbol)
            if stock and 'info' in stock and 'priceInfo' in stock:
                try:
                    suggestions.append({
                        "name": stock["info"]["companyName"],
                        "symbol": symbol,
                        "price": stock["priceInfo"]["lastPrice"],
                        "onedaychange": round(abs(float(stock["priceInfo"]["change"])), 2),
                        "onedaychangepercent": round(abs(float(stock["priceInfo"]["pChange"])), 2),
                        "positive": float(stock["priceInfo"]["pChange"]) > 0
                    })
                except (KeyError, ValueError, TypeError) as e:
                    print(f"Error processing data for {symbol} in getSearchSuggestionsFull: {e}")
            else:
                print(f"Warning: Could not fetch data for {symbol} in getSearchSuggestionsFull")

    return {"suggestions": suggestions}

@app.get("/getTopGainers")
async def getTopGainers():
    # Run blocking call in executor
    top_gainers_df = await run_nse_in_executor(nse.nse_get_top_gainers)
    # Convert DataFrame to dict after getting the result
    top_gainers = top_gainers_df.to_dict(orient="records")
    gainers = []
    for i in top_gainers:
        try: # Add try-except for safety when accessing dict keys
            gainers.append({
                "name": i["meta"]["companyName"],
                "symbol": i["symbol"],
                "price": i["lastPrice"],
                "onedaychange": round(abs(float(i["change"])), 2),
                "onedaychangepercent": round(abs(float(i["pChange"])), 2),
                "positive": float(i["pChange"]) > 0,
                "logo": None
            })
        except (KeyError, ValueError, TypeError) as e:
             print(f"Error processing gainer item: {e}, item: {i}")
    return gainers

@app.get("/getTopLosers")
async def getTopLosers():
    # Run blocking call in executor
    top_losers_df = await run_nse_in_executor(nse.nse_get_top_losers)
    # Convert DataFrame to dict after getting the result
    top_losers = top_losers_df.to_dict(orient="records")
    losers = []
    for i in top_losers:
         try: # Add try-except for safety when accessing dict keys
            losers.append({
                "name": i["meta"]["companyName"],
                "symbol": i["symbol"],
                "price": i["lastPrice"],
                "onedaychange": round(abs(float(i["change"])), 2),
                "onedaychangepercent": round(abs(float(i["pChange"])), 2),
                "positive": float(i["pChange"]) > 0,
                "logo": None
            })
         except (KeyError, ValueError, TypeError) as e:
             print(f"Error processing loser item: {e}, item: {i}")

    return losers

company_websites = {}
try:
    with open("company_websites.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                company_websites[row[0]] = row[1]
except FileNotFoundError:
    print("Warning: company_websites.csv not found.")

@app.get("/getStock")
async def getStock(symbol: str):
    stock = await run_nse_in_executor(nse.nse_eq, symbol)

    if not stock or 'info' not in stock or 'priceInfo' not in stock:
         return JSONResponse(status_code=404, content={"detail": f"Could not retrieve data for symbol {symbol}"})

    logo = None
    website = company_websites.get(symbol)

    try:
        return {
            "name": stock["info"]["companyName"],
            "symbol": symbol,
            "price": stock["priceInfo"]["lastPrice"],
            "onedaychange": round(abs(float(stock["priceInfo"]["change"])), 2),
            "onedaychangepercent": round(abs(float(stock["priceInfo"]["pChange"])), 2),
            "positive": float(stock["priceInfo"]["pChange"]) > 0,
            "website": website,
            "industry": stock.get("industryInfo", {}).get("industry", "N/A"), # Safe access
        }
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error processing stock data for {symbol}: {e}")
        return JSONResponse(status_code=500, content={"detail": f"Error processing data for symbol {symbol}"})

@app.get("/getHoldings")
async def getHoldings(username: str):
    holdings = cur.execute(f"""
        SELECT h.symbol, h.quantity, h.price
        FROM holdings h
        INNER JOIN users u
        ON h.userid = u.userid
        WHERE u.username = ?
    """, (username,)).fetchall()

    holdingsData = {"total":
        {
            "totalInvested": 0,
            "totalCurrent": 0,
            "totalChange": 0,
            "totalChangePercent": 0,
            "positive": False
        }, "holdings": []}

    total_invested = 0
    total_current = 0

    for holding in holdings:
        symbol, quantity, buy_price = holding[0], holding[1], holding[2]
        stock = await run_nse_in_executor(nse.nse_eq, symbol)

        if stock and 'info' in stock and 'priceInfo' in stock:
            try:
                last_price = float(stock["priceInfo"]["lastPrice"])
                totalbuy = buy_price * quantity
                totalcurrent = last_price * quantity

                total_invested += totalbuy
                total_current += totalcurrent

                change = totalcurrent - totalbuy

                holdingsData["holdings"].append({
                "name": stock["info"]["companyName"],
                "symbol": symbol,
                "price": last_price,
                "quantity": quantity,
                "buyPrice": buy_price,
                "currentPrice": last_price,
                "totalBuyPrice": totalbuy,
                "totalCurrentPrice": totalcurrent,
                "change": round(abs(change), 2),
                "changePercent": abs(round((change / totalbuy * 100), 2)) if totalbuy != 0 else 0,
                "positive": change > 0,
                })
            except (KeyError, ValueError, TypeError) as e:
                 print(f"Error processing holding {symbol} for {username}: {e}")
        else:
             print(f"Warning: Could not fetch data for holding {symbol} for user {username}")


    holdingsData["total"]["totalInvested"] = round(total_invested, 2)
    holdingsData["total"]["totalCurrent"] = round(total_current, 2)

    if holdingsData["total"]["totalInvested"] != 0:
        total_change_calc = holdingsData["total"]["totalCurrent"] - holdingsData["total"]["totalInvested"]
        holdingsData["total"]["totalChange"] = round(abs(total_change_calc), 2)
        holdingsData["total"]["totalChangePercent"] = abs(round((total_change_calc / holdingsData["total"]["totalInvested"] * 100), 2))
        holdingsData["total"]["positive"] = total_change_calc > 0

    return holdingsData

## login stuff

@app.get("/checkUsername")
async def checkUsername(query: str):
    cur.execute("SELECT username FROM users WHERE username = ?", (query.strip(),))
    if cur.rowcount > 0:
        return JSONResponse(status_code=400, content={"detail": "Username already taken"})
    else:
        return JSONResponse(status_code=200, content={"detail": "Username available"})

@app.post("/register")
async def register(data: dict):    
    
    password = data["password"]
    username = data["uname"]
    email = data["email"]
    cur.execute("SELECT username FROM users WHERE username = ?", (username,))
    if cur.rowcount > 0:
        return JSONResponse(status_code=400, content={"detail": "Username already taken"})
    hashed_password = get_password_hash(password)
    print(hashed_password)
    cur.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, hashed_password))
    uid = cur.lastrowid
    cur.execute("INSERT INTO wallet (userid) VALUES (?)", (uid,))
    conn.commit()
    return JSONResponse(status_code=201, content={"detail": "Account registered successfully"})


@app.post("/login")
async def login(data: OAuth2PasswordRequestForm = Depends()):
    cur.execute("SELECT userid, password FROM users WHERE username = ?", (data.username,))
    user = cur.fetchall()
    if cur.rowcount==0 or not verify_password(data.password, user[0][1]):
        return JSONResponse(status_code=401, content={"detail": "Incorrect username or password"})
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user[0]}, expires_delta=access_token_expires
    )
    

    unique_key = base64.urlsafe_b64encode(os.urandom(24)).decode('utf-8')
    return {"access_token": access_token, "token_type": "bearer", "category" : "user"}

## wallet endpoints
@app.get("/getWallet")
async def getWallet(username: str):
    cur.execute("SELECT userid FROM users WHERE username = ?", (username,))
    user_result = cur.fetchone()
    if not user_result:
        return JSONResponse(status_code=404, content={"detail": "User not found"})
    user_id = user_result[0]
    cur.execute("SELECT balance FROM wallet WHERE userid = ?", (user_id,))
    balance = cur.fetchall()[0]
    return {"balance": balance}

@app.post("/deposit")
async def deposit(data: dict):
    username = data["username"]
    amount = data["amount"]

    if amount <= 0:
        return JSONResponse(status_code=400, content={"detail": "Amount must be greater than zero"})

    cur.execute("SELECT userid FROM users WHERE username = ?", (username,))
    user = cur.fetchone()
    if not user:
        return JSONResponse(status_code=404, content={"detail": "User not found"})

    user_id = user[0]
    cur.execute("UPDATE wallet SET balance = balance + ? WHERE userid = ?", (amount, user_id))
    conn.commit()
    return {"detail": "Deposit successful"}

@app.post("/withdraw")
async def withdraw(data: dict):
    username = data["username"]
    amount = data["amount"]
    if amount <= 0:
        return JSONResponse(status_code=400, content={"detail": "Amount must be greater than zero"})

    cur.execute("SELECT userid FROM users WHERE username = ?", (username,))
    user = cur.fetchone()
    if not user:
        return JSONResponse(status_code=404, content={"detail": "User not found"})

    user_id = user[0]

    cur.execute("SELECT balance FROM wallet WHERE userid = ?", (user_id,))
    balance = cur.fetchone()[0]
    if balance < amount:
        return JSONResponse(status_code=400, content={"detail": "Insufficient funds"})

    cur.execute("UPDATE wallet SET balance = balance - ? WHERE userid = ?", (amount, user_id))
    conn.commit()
    return {"detail": "Withdrawal successful"}


@app.get("/getQtyOwned")
async def getQtyOwned(username: str, symbol: str):

    try:
        cur.execute("""
            SELECT h.quantity
            FROM holdings h
            INNER JOIN users u ON h.userid = u.userid
            WHERE u.username = ? AND h.symbol = ?
        """, (username, symbol))

        result = cur.fetchone()

        if result:
            quantity = result[0]
        else:
            quantity = 0

        return {"quantity": quantity}

    except Exception as e:
        print(f"Error in getQtyOwned for user {username}, symbol {symbol}: {e}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error fetching quantity"})


# transaction endpoints

@app.post("/buy")
async def buy(data: dict):


    username = data.get("username")
    symbol = data.get("symbol")
    quantity_raw = data.get("quantity")

    if not all([username, symbol, quantity_raw is not None]):
        raise HTTPException(status_code=400, detail="Missing required fields: username, symbol, quantity")

    try:
        quantity = int(quantity_raw)
        if quantity <= 0:
            raise ValueError 
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Quantity must be a positive integer")


    stock_data = await run_nse_in_executor(nse.nse_eq, symbol)
    if not stock_data or "priceInfo" not in stock_data or "lastPrice" not in stock_data["priceInfo"]:
         raise HTTPException(status_code=404, detail=f"Could not retrieve valid price for symbol {symbol}")

    transaction_price = float(stock_data["priceInfo"]["lastPrice"])


    total_cost = transaction_price * quantity

    try:

        cur.execute("SELECT userid FROM users WHERE username = ?", (username,))
        user_result = cur.fetchone()
        if not user_result:
            raise HTTPException(status_code=404, detail=f"User '{username}' not found")
        user_id = user_result[0]

        cur.execute("SELECT balance FROM wallet WHERE userid = ?", (user_id,))
        wallet_result = cur.fetchone()
        if not wallet_result:
             raise HTTPException(status_code=500, detail=f"Wallet data inconsistency for user '{username}'")
        user_balance = wallet_result[0]

        if user_balance < total_cost:
            raise HTTPException(status_code=400, detail=f"Insufficient funds. Required: {total_cost:.2f}, Available: {user_balance:.2f}")

        cur.execute("UPDATE wallet SET balance = balance - ? WHERE userid = ?", (total_cost, user_id))

        cur.execute("SELECT quantity, price FROM holdings WHERE userid = ? AND symbol = ?", (user_id, symbol))
        existing_holding = cur.fetchone()

        if existing_holding:
            old_quantity = existing_holding[0]
            old_avg_price = existing_holding[1]
            new_total_quantity = old_quantity + quantity
            new_avg_price = ((old_quantity * old_avg_price) + (quantity * transaction_price)) / new_total_quantity

            cur.execute("""
                UPDATE holdings
                SET quantity = ?, price = ?
                WHERE userid = ? AND symbol = ?
            """, (new_total_quantity, new_avg_price, user_id, symbol))
            if cur.rowcount == 0:
                raise Exception("Failed to update existing holding during transaction.")
        else:
            cur.execute("""
                INSERT INTO holdings (userid, symbol, quantity, price)
                VALUES (?, ?, ?, ?)
            """, (user_id, symbol, quantity, transaction_price))

        cur.execute("""
            INSERT INTO orders (userid, symbol, quantity, type, price)
            VALUES (?, ?, ?, 'BUY', ?)
        """, (user_id, symbol, quantity, transaction_price))

        conn.commit() 

        return {"detail": f"Successfully bought {quantity} shares of {symbol} at {transaction_price:.2f} each. Total cost: {total_cost:.2f}"}

    except HTTPException as http_exc:
        conn.rollback()
        raise http_exc
    except Exception as e:
        print(f"Transaction Error during buy for user {username}, symbol {symbol}: {e}")
        conn.rollback() 
        raise HTTPException(status_code=500, detail=f"Transaction failed due to server error: {e}")

@app.post("/sell")
async def sell(data: dict):

    username = data.get("username")
    symbol = data.get("symbol")
    quantity_raw = data.get("quantity")

    if not all([username, symbol, quantity_raw is not None]):
        raise HTTPException(status_code=400, detail="Missing required fields: username, symbol, quantity")

    try:
        quantity = int(quantity_raw)
        if quantity <= 0:
            raise ValueError 
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Quantity must be a positive integer")


    stock_data = await run_nse_in_executor(nse.nse_eq, symbol)
    if not stock_data or "priceInfo" not in stock_data or "lastPrice" not in stock_data["priceInfo"]:
         raise HTTPException(status_code=404, detail=f"Could not retrieve valid price for symbol {symbol}")

    transaction_price = float(stock_data["priceInfo"]["lastPrice"])


    total_revenue = transaction_price * quantity

    try:

        cur.execute("SELECT userid FROM users WHERE username = ?", (username,))
        user_result = cur.fetchone()
        if not user_result:
            raise HTTPException(status_code=404, detail=f"User '{username}' not found")
        user_id = user_result[0]

        cur.execute("SELECT balance FROM wallet WHERE userid = ?", (user_id,))
        wallet_result = cur.fetchone()
        if not wallet_result:
             raise HTTPException(status_code=500, detail=f"Wallet data inconsistency for user '{username}'")
        user_balance = wallet_result[0]

        cur.execute("SELECT quantity, price FROM holdings WHERE userid = ? AND symbol = ?", (user_id, symbol))
        existing_holding = cur.fetchone()

        if not existing_holding:
            raise HTTPException(status_code=400, detail=f"You don't have any {symbol} shares to sell")

        old_quantity = existing_holding[0]
        old_avg_price = existing_holding[1]

        if old_quantity < quantity:
            raise HTTPException(status_code=400, detail=f"You don't have enough {symbol} shares to sell")

        new_quantity = old_quantity - quantity
        new_avg_price = old_avg_price

        if new_quantity == 0:
            cur.execute("DELETE FROM holdings WHERE userid = ? AND symbol = ?", (user_id, symbol))
        else:
            cur.execute("""
                UPDATE holdings
                SET quantity = ?, price = ?
                WHERE userid = ? AND symbol = ?
            """, (new_quantity, new_avg_price, user_id, symbol))
        if cur.rowcount == 0:
            raise Exception("Failed to update existing holding during transaction.")

        cur.execute("UPDATE wallet SET balance = balance + ? WHERE userid = ?", (total_revenue, user_id))

        cur.execute("""
            INSERT INTO orders (userid, symbol, quantity, type, price)
            VALUES (?, ?, ?, 'SELL', ?)
        """, (user_id, symbol, quantity, transaction_price))

        conn.commit() 

        return {"detail": f"Successfully sold {quantity} shares of {symbol} at {transaction_price:.2f} each. Total revenue: {total_revenue:.2f}"}

    except HTTPException as http_exc:
        conn.rollback()
        raise http_exc
    except Exception as e:
        print(f"Transaction Error during sell for user {username}, symbol {symbol}: {e}")
        conn.rollback() 
        raise HTTPException(status_code=500, detail=f"Transaction failed due to server error: {e}")


prediction_cache = {}

@app.get("/predict/{symbol}")
async def predict_stock(symbol: str):
    symbol = symbol.strip().upper()
    today = date.today()
    logging.info(f"Received prediction request for: {symbol}")

    cached_result = prediction_cache.get(symbol)
    if cached_result:
        generation_date = cached_result.get("generation_date")
        if generation_date == today:
            logging.info(f"Cache hit for {symbol} generated on {today.isoformat()}. Returning cached data.")
            return cached_result["data"]
        else:
            logging.info(f"Cache exists for {symbol} but is stale (generated on {generation_date.isoformat()}). Will regenerate.")
    else:
        logging.info(f"Cache miss for {symbol}. Generating new prediction.")

    results = get_stock_predictions(symbol, days_to_predict=7)

    if results and 'predictions' in results and 'last_date' in results and 'filename' in results:
        logging.info(f"Successfully generated new prediction for {symbol}.")

        last_date_str = results['last_date'].strftime('%Y-%m-%d')
        api_response_data = {
            "symbol": symbol,
            "last_historical_date": last_date_str,
            "predicted_prices": results['predictions'],
            "filename": results['filename'],
            "shap_values": results.get('shap_values'),
            "lime_weights": results.get('lime_weights')
        }

        prediction_cache[symbol] = {
            "generation_date": today,
            "data": api_response_data 
        }
        logging.info(f"Stored prediction for {symbol} in cache for {today.isoformat()}.")

        return api_response_data
    else:
        logging.error(f"Prediction generation failed for symbol {symbol}.")
        raise HTTPException(status_code=500, detail=f"Failed to generate prediction for {symbol}")
