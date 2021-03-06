# trading_calendars
A Python library of exchange calendars meant to be used with [Zipline](https://github.com/quantopian/zipline).

| Exchange                        | ISO Code | Country     | Version Added | Exchange Website (English)                              |
| ------------------------------- | -------- | ----------- | ------------- | ------------------------------------------------------- |
| New York Stock Exchange         | XNYS     | USA         | 1.0           | https://www.nyse.com/index                              |
| CBOE Futures                    | XCBF     | USA         | 1.0           | https://markets.cboe.com/us/futures/overview/           |
| Chicago Mercantile Exchange     | CMES     | USA         | 1.0           | https://www.cmegroup.com/                               |
| ICE US                          | IEPA     | USA         | 1.0           | https://www.theice.com/index                            |
| Toronto Stock Exchange          | XTSE     | Canada      | 1.0           | https://www.tsx.com/                                    |
| BMF Bovespa                     | BVMF     | Brazil      | 1.0           | http://www.b3.com.br/en_us/                             |
| Euronext Amsterdam              | XAMS     | Netherlands | 1.2           | https://www.euronext.com/en/regulation/amsterdam        |
| Euronext Brussels               | XBRU     | Belgium     | 1.2           | https://www.euronext.com/en/regulation/brussels         |
| Euronext Lisbon                 | XLIS     | Portugal    | 1.2           | https://www.euronext.com/en/regulation/lisbon           |
| Euronext Paris                  | XPAR     | France      | 1.2           | https://www.euronext.com/en/regulation/paris            |
| Frankfurt Stock Exchange        | XFRA     | Germany     | 1.2           | http://en.boerse-frankfurt.de/                          |
| SIX Swiss Exchange              | XSWX     | Switzerland | 1.2           | https://www.six-group.com/exchanges/index.html          |
| Tokyo Stock Exchange            | XTKS     | Japan       | 1.2           | https://www.jpx.co.jp/english/                          |
| Austrialian Securities Exchange | XASX     | Australia   | 1.3           | https://www.asx.com.au/                                 |
| Bolsa de Madrid                 | XMAD     | Spain       | 1.3           | http://www.bolsamadrid.es/ing/aspx/Portada/Portada.aspx |
| Borsa Italiana                  | XMIL     | Italy       | 1.3           | https://www.borsaitaliana.it/homepage/homepage.en.htm   |
| New Zealand Exchange            | XNZE     | New Zealand | 1.3           | https://www.nzx.com/                                    |
| Wiener Borse                    | XWBO     | Austria     | 1.3           | https://www.wienerborse.at/en/                          |
| Hong Kong Stock Exchange        | XHKG     | Hong Kong   | 1.3           | https://www.hkex.com.hk/?sc_lang=en                     |
| Copenhagen Stock Exchange       | XCSE     | Denmark     | 1.4           | http://www.nasdaqomxnordic.com/                         |
| Helsinki Stock Exchange         | XHEL     | Finland     | 1.4           | http://www.nasdaqomxnordic.com/                         |
| Stockholm Stock Exchange        | XSTO     | Sweden      | 1.4           | http://www.nasdaqomxnordic.com/                         |
| Oslo Stock Exchange             | XOSL     | Norway      | 1.4           | https://www.oslobors.no/ob_eng/                         |
| Irish Stock Exchange            | XDUB     | Ireland     | 1.4           | http://www.ise.ie/                                      |
| Bombay Stock Exchange           | XBOM     | India       | 1.5           | https://www.bseindia.com                                |
| Singapore Exchange              | XSES     | Singapore   | 1.5           | https://www.sgx.com                                     |
| Shanghai Stock Exchange         | XSHG     | China       | 1.5           | http://english.sse.cnn.cn                               |
| Korea Exchange                  | XKRX     | South Korea | 1.6           | http://global.krx.co.kr                                 |

Calendars marked with an asterisk (*) have not yet been released.

Note that exchange calendars are defined by their [ISO-10383 market identifier code](https://www.iso20022.org/10383/iso-10383-market-identifier-codes).

## Usage
```python
from trading_calendars import get_calendar

# 不考虑午休时间，日历期间为1990-12-19 ~ 2025-12-31
# A股交易日 (上交所、深交所)
calendar = get_calendar('XSHG')

calendar.actual_last_session
```

## Install
在windows 10中python 3.7环境下安装`lru-dict`,[下载](https://www.lfd.uci.edu/~gohlke/pythonlibs/#lru_dict)后使用`pip install lru*.whl`