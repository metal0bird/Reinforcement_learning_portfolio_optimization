var url = 'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords=<SYMBOL>&apikey=QPKZV3VMXHXVHDX3';

const options = {
    method: 'GET',
    headers: { 'User-Agent': 'request' },
    json: true
};

const getStockSymbol = async (search) => {
    const res = await fetch(url.replace('<SYMBOL>', search), options);
    const data = await res.json();
    return data['bestMatches'][0]['1. symbol'];

};

export default async function handler(req, res) {
    const data = await getStockSymbol(req.query.search);
    res.status(200).json(data);
}