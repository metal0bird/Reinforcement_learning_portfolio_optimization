var url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=<SYMBOL>&interval=15min&outputsize=full&apikey=QPKZV3VMXHXVHDX3&datatype=csv';

const options = {
  method: 'GET',
  headers: { 'User-Agent': 'request' },
  json: true
};

const getStockData = async (symbol) => {
  const res = await fetch(url.replace('<SYMBOL>', symbol), options);
  const data = await res.text();
  return data;
};

export default async function handler(req, res) {
  const data = await getStockData(req.query.symbol);
  res.status(200).json(data);
}