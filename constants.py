# constants.py
# This file centralizes all static configuration data for the application.

import plotly.express as px

# Dictionary mapping sectors to their respective stock tickers.
SECTORS = {
    'Technology': ['MSFT', 'NVDA', 'AVGO', 'CRM', 'AMD', 'QCOM', 'ORCL', 'ADBE', 'ACN', 'INTC', 'IBM', 'CSCO', 'AMAT', 'UBER', 'NOW', 'INTU', 'LRCX', 'APH', 'ADI', 'PANW', 'MU', 'SNPS', 'CDNS', 'KLAC', 'ROP', 'DELL', 'ANET', 'MCHP', 'FTNT', 'TEL', 'IT', 'CTSH', 'SNOW', 'MSI', 'PAYX', 'ADSK', 'KEYS', 'CRWD', 'GLW', 'HPE', 'HPQ', 'CDW', 'FICO', 'TER', 'STX', 'ANSS', 'TRMB', 'VRSN', 'ZBRA', 'ENET', 'TDY', 'WDC', 'JBL', 'EXC', 'UI', 'NTAP', 'SMCI', 'GPN', 'PTC', 'FFIV', 'FLT', 'GRMN', 'DXC', 'FIS', 'AKAM', 'KEYS', 'BR', 'MDB', 'DDOG', 'NET', 'TEAM', 'ZS', 'OKTA', 'PLTR', 'SHOP', 'SQ', 'ARM', 'WDAY', 'HUBS', 'ZM', 'EPAM', 'APP', 'AFRM', 'IOT', 'S', 'GEN', 'CHKP', 'DT', 'VEEV', 'FTV', 'TOST', 'DOCN', 'UIPATH', 'PATH', 'AI', 'Z', 'GTLB', 'RBLX', 'MSTR'],
    'Consumer Cyclical': ['IREN','AMZN', 'TSLA', 'HD', 'MCD', 'LOW', 'SBUX', 'BKNG', 'TJX', 'MAR', 'CMG', 'GM', 'F', 'ROST', 'HLT', 'YUM', 'ORLY', 'AZO', 'LVS', 'CPRT', 'NKE', 'CCL', 'RCL', 'DRI', 'GRMN', 'ULTA', 'LEN', 'DHI', 'TGT', 'EBAY', 'EXPE', 'POOL', 'WYN', 'PHM', 'KMX', 'BBY', 'WYNN', 'MGM', 'ETSY', 'LULU', 'TPR', 'CZR', 'NVR', 'DPZ', 'RL', 'VFC', 'WHR', 'HAS', 'MAT', 'MHK', 'APTV', 'GPC', 'LKQ', 'BWA', 'KDP', 'MNST', 'HOG', 'PENN', 'DKS', 'FL', 'BBWI', 'RH', 'WYND', 'CHS', 'AAP', 'LEG', 'GPS', 'JWN', 'M', 'KSS', 'TCOM', 'TRIP', 'SNA', 'SWK', 'PAG', 'AN', 'RUSHA', 'SAH', 'ABG', 'CWH', 'GNTX', 'LEA', 'VC', 'SON', 'TPX', 'WGO', 'THO'],
    'Financials': ['BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'AXP', 'SPGI', 'SCHW', 'CB', 'PNC', 'C', 'AON', 'MMC', 'USB', 'ICE', 'TROW', 'COF', 'MET', 'PRU', 'AIG', 'TRV', 'ALL', 'CME', 'DFS', 'PGR', 'WRB', 'MCO', 'PYPL', 'FISV', 'SQ', 'ACGL', 'RE', 'AMP', 'AJG', 'BEN', 'IVZ', 'L', 'NTRS', 'STT', 'TFC', 'ZION', 'FITB', 'KEY', 'HBAN', 'RF', 'CFG', 'CMA', 'MTB', 'BAC', 'GS', 'MS', 'JPM', 'C', 'WFC', 'USB', 'PNC', 'TFC', 'COF', 'BK', 'STT', 'NTRS', 'AMP', 'AXP', 'DFS', 'MA', 'V', 'PYPL', 'FISV', 'GPN', 'FLT', 'SQ', 'AFRM', 'HOOD', 'COIN', 'CBOE', 'NDAQ', 'MKTX', 'TRU', 'EFX', 'FDS', 'AFL', 'CINF', 'HIG', 'LNC', 'UNM'],
    'Energy': ['XOM', 'CVX', 'SHEL', 'COP', 'SLB', 'EOG', 'PXD', 'OXY', 'MPC', 'VLO', 'WMB', 'KMI', 'PSX', 'HAL', 'DVN', 'HES', 'OKE', 'BKR', 'FANG', 'TRGP', 'CTRA', 'MRO', 'APA', 'PBR', 'EQT', 'ENB', 'TRP', 'SU', 'CNQ', 'IMO', 'CVE', 'PBA', 'ET', 'EPD', 'PAA', 'MPLX', 'LNG', 'NOV', 'FTI', 'OII', 'HP', 'RIG', 'VET', 'MUR', 'NFG', 'AR', 'RRC', 'SWN', 'CHK', 'CNX', 'ARCH', 'BTU', 'CEIX', 'METC', 'WFRD', 'DK', 'PBF', 'CVI', 'USAC', 'NGL', 'SUN', 'GEL', 'CQP', 'SHLS', 'ENPH', 'SEDG', 'RUN', 'FSLR', 'SPWR', 'BE', 'PLUG', 'FCEL', 'BLDP'],
    'Healthcare': ['LLY', 'UNH', 'JNJ', 'MRK', 'ABBV', 'TMO', 'PFE', 'DHR', 'ABT', 'AMGN', 'MDT', 'GILD', 'ISRG', 'SYK', 'CVS', 'REGN', 'VRTX', 'CI', 'BSX', 'BDX', 'HCA', 'ELV', 'MCK', 'ZBH', 'HUM', 'IDXX', 'EW', 'BIIB', 'COR', 'LH', 'ALGN', 'A', 'DXCM', 'RMD', 'WST', 'TECH', 'IQV', 'HOLX', 'DGX', 'CNC', 'BAX', 'CAH', 'MOH', 'UHS', 'XRAY', 'INCY', 'SGEN', 'JAZZ', 'BGNE', 'MRNA', 'BNTX', 'NVAX', 'VEEV', 'TDOC', 'CERT', 'DVA', 'PODD', 'TFX', 'COO', 'STE', 'SRPT'],
    'Industrials': ['CAT', 'UPS', 'GE', 'UNP', 'BA', 'RTX', 'HON', 'DE', 'LMT', 'ADP', 'ETN', 'WM', 'CSX', 'NSC', 'PCAR', 'ITW', 'GD', 'TDG', 'EMR', 'ROP', 'PAYX', 'CMI', 'PH', 'VRSK', 'FAST', 'OTIS', 'CARR', 'GWW', 'JCI', 'RSG', 'IR', 'AME', 'LDOS', 'XYL', 'WAB', 'EXPD', 'JBHT', 'ODFL', 'CHRW', 'DAL', 'UAL', 'AAL', 'LUV', 'FDX', 'SWK', 'SNA', 'DOV', 'PNR', 'ROK', 'TT', 'GNRC', 'MMM', 'TXT', 'HII', 'NOC', 'LHX', 'HWM', 'SPR', 'MAS', 'FBHS', 'BLD', 'J', 'KBR', 'FLR', 'PWR', 'FIX', 'URI', 'AIT', 'AVY', 'SIRI'],
    'Consumer Defensive': ['WMT', 'PG', 'COST', 'KO', 'PEP', 'MDLZ', 'MO', 'PM', 'TGT', 'CL', 'EL', 'KMB', 'GIS', 'KR', 'ADM', 'SYY', 'KDP', 'STZ', 'MNST', 'DG', 'DLTR', 'HSY', 'K', 'CPB', 'CAG', 'HRL', 'SJM', 'MKC', 'BF-B', 'CHD', 'TAP', 'TSN', 'CALM', 'LANC', 'SFBS', 'UNFI', 'HFFG', 'THS', 'JJSF', 'BGS', 'FLO', 'PPC', 'LW', 'POST', 'BRBR', 'UTZ', 'PRMW', 'BYND', 'CELH', 'CHGG', 'EDG', 'HELE', 'HRB', 'NWL', 'REV', 'CUTR', 'PRPL', 'TPX', 'SCI', 'CSV', 'MATW'],
    'Communication Services': ['GOOGL', 'META', 'GOOG', 'DIS', 'NFLX', 'CMCSA', 'TMUS', 'VZ', 'T', 'CHTR', 'ATVI', 'EA', 'WBD', 'IPG', 'OMC', 'LYV', 'TTWO', 'PARA', 'FOXA', 'DISCA', 'ROKU', 'MTCH', 'TME', 'SPOT', 'SNAP', 'PINS', 'TWTR', 'BILI', 'Z', 'ZG', 'IAC', 'NWSA', 'NYT', 'GTN', 'SBGI', 'TGNA', 'VIAC', 'FWONA', 'LSXMA', 'SIRI', 'DISCK', 'DISH', 'LUMN', 'FYBR', 'ATEX', 'IDT', 'MAX', 'MGNI', 'PUBM', 'TTD', 'PERI', 'APPS', 'UPLD'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'PEG', 'WEC', 'EIX', 'ES', 'AWK', 'ED', 'PCG', 'DTE', 'FE', 'AEE', 'ETR', 'PPL', 'CNP', 'SJI', 'LNT', 'CMS', 'NI', 'EVRG', 'PNW', 'NRG', 'AES', 'ALE', 'BKH', 'CPK', 'IDA', 'MGEE', 'OGE', 'OTTR', 'WTRG'],
    'Real Estate': ['PLD', 'AMT', 'EQIX', 'CCI', 'O', 'PSA', 'SPG', 'WELL', 'DLR', 'AVB', 'WY', 'EQR', 'SBAC', 'VICI', 'ARE', 'INVH', 'VTR', 'ESS', 'MAA', 'CPT', 'EXR', 'CBRE', 'UDR', 'IRM', 'KIM', 'REG', 'FRT', 'PEAK', 'BXP', 'HST', 'MAC', 'SLG', 'VNO', 'DEA', 'KRC', 'HIW', 'SRC', 'ADC', 'EPRT', 'NNN', 'WPC', 'GLPI', 'LAMR', 'OUT', 'PCH', 'GEO', 'CXW', 'UNIT', 'LSI', 'RITM', 'STWD', 'ARI', 'ACRE', 'BRT', 'GOOD', 'IRT', 'APTS', 'AIRC', 'STAG', 'TRNO', 'FR', 'WSR', 'CIO', 'DEA'],
    'Basic Materials': ['LIN','RIO','NEM','SHW','AEM','ECL','BHP','APD','B','WPM','VALE','FNV','GFI','AU','DD','KGC','PPG','TECK','IFF','LYB','RPM','AGI','MP','RGLD','CDE','HMY','SQM','ALB','WLK','AVTR','EQX','HL','SBSW','NEU','IAG','OR','BTG','EMN','ESI','AXTA','EGO','NGD','BCPC','SSRM','NG','SXT','CBT','TMC','SSL','ORLA','SAND','HWKN','USAR','PRM','FUL','USAS','AVNT','WDFC','OLN','SA','DRD','MTRN','CGAU','KWR','ASH','SKE','AAUC','NGVT','CC','LAC','IOSP','MTX','IPX','ARMN','VZLA','UAMY','CRML','CNL','NAK','SCL','REX','TMQ','ECVT','GSM','SLI','ODC','NB','CMP','IAUX','LAR','GAU','SGML','NEXA','LWLG','CMCL','IDR','GRPE','KRO','GTI','NFGC']
}

# Dictionary mapping index tickers to their full names for display purposes.
INDEX_TICKER_TO_NAME = {
    'XLK': 'US Tech Giants (XLK)', 'SOXX': 'Semiconductors (SOXX)',
    'SKYY': 'Cloud Computing (SKYY)', 'XLF': 'Financials (XLF)',
    'XLV': 'Healthcare (XLV)', 'XRT': 'US Retail (XRT)',
    'HERO': 'Gaming (HERO)', 'XLE': 'Energy (XLE)',
    'XLI': 'Industrials (XLI)', 'XLP': 'Consumer Staples (XLP)',
    'XLC': 'Communication Services (XLC)', 'XLU': 'Utilities (XLU)',
    'XLRE': 'Real Estate (XLRE)', 'XLB': 'Basic Materials (XLB)',
    '^GSPC': 'S&P 500 Index', '^NDX': 'NASDAQ 100 Index'
}

# Dictionary mapping sectors to relevant benchmark indices.
SECTOR_TO_INDEX_MAPPING = {
    'Technology': ['XLK', 'SOXX', 'SKYY'],
    'Financials': ['XLF'],
    'Healthcare': ['XLV'],
    'Consumer Cyclical': ['XRT', 'HERO'],
    'Energy': ['XLE'],
    'Industrials': ['XLI'],
    'Consumer Defensive': ['XLP'],
    'Communication Services': ['XLC'],
    'Utilities': ['XLU'],
    'Real Estate': ['XLRE'],
    'Basic Materials': ['XLB']
}

# Generate a list of all possible symbols for color mapping
all_possible_symbols = list(INDEX_TICKER_TO_NAME.keys())
for sector_tickers in SECTORS.values():
    all_possible_symbols.extend(sector_tickers)
all_possible_symbols = sorted(list(set(all_possible_symbols)))

# Pre-defined color map for consistent chart coloring across the app.
COLOR_DISCRETE_MAP = {
    symbol: color for symbol, color in zip(
        all_possible_symbols,
        px.colors.qualitative.Plotly * (len(all_possible_symbols) // 10 + 1)
    )
}