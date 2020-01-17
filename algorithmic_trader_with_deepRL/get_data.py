import os
import pandas as pd
import asyncio
import aiohttp
import aiofiles
import datetime as dt
from dotenv import load_dotenv


async def query_api(symbol):
	load_dotenv()

	base_url = 'https://www.alphavantage.co'
	endpoint = '/query'
	url = f'{base_url}{endpoint}'
	params = {
		'function': 'TIME_SERIES_DAILY_ADJUSTED',
		'symbol': symbol,
		'outputsize': 'full',
		'apikey': os.getenv('ALPHAVANTAGE_API'),
		'datatype': 'csv'
	}
	print(f'Querying API for {symbol} data... \n')
	async with aiohttp.ClientSession() as session:
		async with session.get(url, params=params) as resp:
			data = await resp.text()
	async with aiofiles.open(f'./data/daily_adjusted_{symbol}.csv', 'w') as outfile:
		await outfile.write(data)

def prompt_csv():
	print('Enter Symbol:', end =" ")
	symbol = input()
	loop = asyncio.get_event_loop()
	loop.run_until_complete(query_api(symbol))
	print('Done!')

def get_csv(symbol):
	loop = asyncio.get_event_loop()
	loop.run_until_complete(query_api(symbol))

def get_path(symbol, base_dir=None):
	if base_dir is None:
		base_dir = os.environ.get('PWD')
	return os.path.join(base_dir, f'data/daily_adjusted_{symbol}.csv')


def load_data(symbols, dates, SPY=True, column_name='adjusted_close'):
	df = pd.DataFrame(index = dates)

	if SPY and 'SPY' not in symbols: symbols.append('SPY')

	for symbol in symbols:
		temp = pd.read_csv(get_path(symbol), index_col = 'timestamp', parse_dates=True, usecols=['timestamp', column_name], na_values=['nan'])
		temp.rename(columns={f'{column_name}': symbol}, inplace=True)
		df = df.join(temp)
	if symbol == 'SPY': df = df.dropna(subset=["SPY"])
	return df

if __name__ == "__main__":
	prompt_csv()
	# df = load_data(['GOOG'], dates=pd.date_range(dt.datetime(2018,1,1), dt.datetime(2018,12,31)))
