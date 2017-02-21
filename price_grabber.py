import datetime, json, random
try:
    from urllib.request import build_opener
except:
    from urllib2 import build_opener

# Makes a request to a given URL (first arg) and optional params (second arg)
def make_request(*args):
    opener = build_opener()
    opener.addheaders = [('User-agent',
                          'Mozilla/5.0'+str(random.randrange(1000000)))]
    try: 
        return opener.open(*args).read().strip()
    except Exception as e:
        try:
            p = e.read().strip()
        except:
            p = e
        raise Exception(p)

coins = ['BTC', 'ETH', 'LTC', 'XMR', 'XRP', 'DASH', 'MAID', 'XEM', 'DOGE']
altcoins = coins[1:]

now = int(datetime.datetime.now().strftime('%s'))
prices = []
coinstring = ','.join(altcoins)
for t in range(now, now - 86400 * 365, -86400):
    btcprice = json.loads(make_request('https://min-api.cryptocompare.com/data/pricehistorical?fsym=BTC&tsyms=USD&ts=%d' % t))["BTC"]["USD"]
    cryptoprices = json.loads(make_request('https://min-api.cryptocompare.com/data/pricehistorical?fsym=BTC&tsyms=%s&ts=%d' % (coinstring, t)))["BTC"]
    pricelist = {'BTC': btcprice, 'timestamp': t}
    for p in altcoins:
        pricelist[p] = btcprice * 1.0 / cryptoprices[p]
    print pricelist
    prices.append(pricelist)

print 'all prices', prices

cov = {}
for k1 in coins:
    cov[k1] = {}
    for k2 in coins:
        cov[k1][k2] = 0
for i in range(len(prices) - 1):
    for k1 in coins:
        pricechangek1 = prices[i + 1][k1] / prices[i][k1] - 1
        for k2 in coins:
            pricechangek2 = prices[i + 1][k2] / prices[i][k2] - 1
            cov[k1][k2] += pricechangek1 * pricechangek2
print 'raw cov', cov
var = {c: cov[c][c] for c in coins}
for k1 in coins:
    for k2 in coins:
        cov[k1][k2] /= (var[k1] ** 0.5 * var[k2] ** 0.5)
print 'correlation', cov
