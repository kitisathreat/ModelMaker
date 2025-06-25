import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import warnings
import os
warnings.filterwarnings('ignore')

class SECFileSourcer:
    """
    A comprehensive class for sourcing SEC filings and creating financial models.
    
    This class provides methods to:
    1. Find and download SEC 10-K and 10-Q filings
    2. Create traditional financial three-statement models
    3. Generate sensitivity analysis with operating leverage impacts
    4. Create KPI summary sheets
    """
    
    # Class-level cache for ticker-to-CIK mapping
    _ticker_cik_cache = None
    _cache_last_updated = None
    _cache_expiry_seconds = 24 * 3600  # 24 hours
    
    def __init__(self):
        """Initialize the SEC File Sourcer with base URLs and headers."""
        self.base_url = "https://data.sec.gov"
        self.sec_ticker_url = "https://www.sec.gov/files/company_tickers.json"
        self.headers = {
            'User-Agent': 'ModelMaker/1.0 (kit.kumar@gmail.com)',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Add rate limiting - SEC requires delays between requests
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms minimum between requests
        
    def _rate_limit(self):
        """Ensure minimum time between API requests to comply with SEC rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    @classmethod
    def _is_cache_valid(cls):
        if cls._ticker_cik_cache is None or cls._cache_last_updated is None:
            return False
        return (time.time() - cls._cache_last_updated) < cls._cache_expiry_seconds

    @classmethod
    def _load_ticker_cik_cache(cls, session, url):
        """Load the ticker-to-CIK mapping from the SEC and cache it."""
        try:
            response = session.get(url)
            if response.status_code != 200:
                print(f"Error fetching company tickers: {response.status_code}")
                return None
            data = response.json()
            # The SEC file is now a dictionary with numeric string keys
            cache = {}
            for key, entry in data.items():
                if isinstance(entry, dict) and 'ticker' in entry and 'cik_str' in entry:
                    cache[entry['ticker'].upper()] = str(entry['cik_str']).zfill(10)
            cls._ticker_cik_cache = cache
            cls._cache_last_updated = time.time()
            return cache
        except Exception as e:
            print(f"Error loading ticker-to-CIK cache: {e}")
            return None

    def get_cik_from_ticker(self, ticker: str, force_refresh: bool = False) -> Optional[str]:
        """
        Convert a stock ticker symbol to its corresponding CIK (Central Index Key) number.
        Uses a cached mapping for efficiency.
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            force_refresh (bool): If True, refresh the cache from the SEC
        Returns:
            Optional[str]: CIK number as a string (10-digit zero-padded), or None if not found
        """
        ticker_upper = ticker.upper()
        # Check cache
        if force_refresh or not self._is_cache_valid():
            self._rate_limit()
            cache = self._load_ticker_cik_cache(self.session, self.sec_ticker_url)
            if cache is None:
                print("Could not fetch ticker-to-CIK mapping from SEC.")
                return None
        else:
            cache = self._ticker_cik_cache
        cik = cache.get(ticker_upper)
        if cik:
            return cik
        print(f"Ticker '{ticker}' not found in SEC database.")
        return None
    
    def find_sec_filings(self, ticker: str, filing_types: List[str] = ['10-K', '10-Q']) -> pd.DataFrame:
        """
        Find SEC 10-K and 10-Q filings for a given stock ticker, sorted by date.
        
        Args:
            ticker (str): Stock ticker symbol
            filing_types (List[str]): Types of filings to search for (default: ['10-K', '10-Q'])
            
        Returns:
            pd.DataFrame: DataFrame containing filing information sorted by date
        """
        try:
            # Convert ticker to CIK
            cik = self.get_cik_from_ticker(ticker)
            if not cik:
                print(f"Could not find CIK for ticker: {ticker}")
                return pd.DataFrame()
            
            # Get company submissions
            submissions_url = f"{self.base_url}/submissions/CIK{cik}.json"
            response = self.session.get(submissions_url)
            
            if response.status_code != 200:
                print(f"Error fetching submissions for {ticker} (CIK: {cik}): {response.status_code}")
                return pd.DataFrame()
            
            submissions_data = response.json()
            filings = submissions_data.get('filings', {}).get('recent', {})
            
            # Create DataFrame from filings
            filing_data = []
            for i in range(len(filings.get('form', []))):
                form = filings['form'][i]
                if form in filing_types:
                    filing_data.append({
                        'form': form,
                        'filingDate': filings['filingDate'][i],
                        'accessionNumber': filings['accessionNumber'][i],
                        'primaryDocument': filings['primaryDocument'][i],
                        'description': filings.get('description', [''])[i] if i < len(filings.get('description', [])) else ''
                    })
            
            df = pd.DataFrame(filing_data)
            if not df.empty:
                df['filingDate'] = pd.to_datetime(df['filingDate'])
                df = df.sort_values('filingDate', ascending=False)
                
            return df
            
        except Exception as e:
            print(f"Error in find_sec_filings: {str(e)}")
            return pd.DataFrame()
    
    def create_financial_model(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Create a traditional financial three-statement model with annual and quarterly views.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing financial model dataframes
        """
        try:
            # Initialize the financial model structure
            financial_model = {
                'annual_income_statement': pd.DataFrame(),
                'annual_balance_sheet': pd.DataFrame(),
                'annual_cash_flow': pd.DataFrame(),
                'quarterly_income_statement': pd.DataFrame(),
                'quarterly_balance_sheet': pd.DataFrame(),
                'quarterly_cash_flow': pd.DataFrame()
            }
            
            # Convert ticker to CIK
            cik = self.get_cik_from_ticker(ticker)
            if not cik:
                print(f"Could not find CIK for ticker: {ticker}")
                return financial_model
            
            # Remove leading zeros from CIK for new endpoint
            cik_no_zeros = str(int(cik))
            # Get company facts from new endpoint
            company_facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_no_zeros}.json"
            response = self.session.get(company_facts_url)
            
            if response.status_code == 200:
                facts_data = response.json()
                facts = facts_data.get('facts', {})
                # Define key financial metrics for each statement
                income_statement_metrics = {
                    'Revenues': 'us-gaap:Revenues',
                    'CostOfRevenue': 'us-gaap:CostOfRevenue',
                    'GrossProfit': 'us-gaap:GrossProfit',
                    'OperatingExpenses': 'us-gaap:OperatingExpenses',
                    'OperatingIncomeLoss': 'us-gaap:OperatingIncomeLoss',
                    'NetIncomeLoss': 'us-gaap:NetIncomeLoss',
                    'EarningsPerShareBasic': 'us-gaap:EarningsPerShareBasic',
                    'EarningsPerShareDiluted': 'us-gaap:EarningsPerShareDiluted'
                }
                balance_sheet_metrics = {
                    'CashAndCashEquivalents': 'us-gaap:CashAndCashEquivalentsAtCarryingValue',
                    'TotalAssets': 'us-gaap:Assets',
                    'TotalCurrentAssets': 'us-gaap:AssetsCurrent',
                    'TotalLiabilities': 'us-gaap:Liabilities',
                    'TotalCurrentLiabilities': 'us-gaap:LiabilitiesCurrent',
                    'TotalStockholdersEquity': 'us-gaap:StockholdersEquity',
                    'TotalDebt': 'us-gaap:LongTermDebtNoncurrent'
                }
                cash_flow_metrics = {
                    'NetCashProvidedByUsedInOperatingActivities': 'us-gaap:NetCashProvidedByUsedInOperatingActivities',
                    'NetCashProvidedByUsedInInvestingActivities': 'us-gaap:NetCashProvidedByUsedInInvestingActivities',
                    'NetCashProvidedByUsedInFinancingActivities': 'us-gaap:NetCashProvidedByUsedInFinancingActivities',
                    'CapitalExpenditures': 'us-gaap:PaymentsToAcquirePropertyPlantAndEquipment',
                    'DividendsPaid': 'us-gaap:PaymentsOfDividends'
                }
                for period in ['annual', 'quarterly']:
                    for statement, metrics in [('income_statement', income_statement_metrics),
                                             ('balance_sheet', balance_sheet_metrics),
                                             ('cash_flow', cash_flow_metrics)]:
                        df = self._extract_financial_data(facts, metrics, period)
                        financial_model[f'{period}_{statement}'] = df
                return financial_model
            else:
                print(f"XBRL JSON not available, trying XBRL XML instance document for {ticker} (CIK: {cik_no_zeros})")
                # Fallback: Try to find and parse XBRL XML instance document using Arelle
                try:
                    filings_df = self.find_sec_filings(ticker)
                    if filings_df.empty:
                        print(f"No filings found for {ticker}")
                        return financial_model
                    # Try the most recent 10-K or 10-Q
                    for idx, row in filings_df.iterrows():
                        accession_number = row['accessionNumber']
                        cik_dir = str(int(cik))
                        accession_clean = accession_number.replace('-', '')
                        filing_dir_url = f"https://www.sec.gov/Archives/edgar/data/{cik_dir}/{accession_clean}/"
                        # List files in the directory
                        dir_response = self.session.get(filing_dir_url)
                        if dir_response.status_code == 200:
                            # Find .xml files (XBRL instance docs)
                            import re
                            xml_files = re.findall(r'href="([^"]+\.xml)"', dir_response.text)
                            xbrl_instance_file = None
                            # Prefer files ending with _htm.xml and not FilingSummary.xml
                            for xml_file in xml_files:
                                if xml_file.endswith('_htm.xml') and 'FilingSummary' not in xml_file:
                                    xbrl_instance_file = xml_file
                                    break
                            if xbrl_instance_file is None:
                                # Fallback: pick the first .xml file that's not FilingSummary
                                for xml_file in xml_files:
                                    if 'FilingSummary' not in xml_file:
                                        xbrl_instance_file = xml_file
                                        break
                            if xbrl_instance_file:
                                xbrl_url = f"https://www.sec.gov/Archives/edgar/data/{cik_dir}/{accession_clean}/{xbrl_instance_file}"
                                print(f"Parsing XBRL instance document: {xbrl_url}")
                                # Use Arelle to parse the XBRL instance document
                                try:
                                    from arelle import Cntlr
                                    cntlr = Cntlr.Cntlr(logFileName=None)
                                    model_xbrl = cntlr.modelManager.load(xbrl_url)
                                    
                                    # Extract facts from the XBRL instance document
                                    facts_from_xml = {}
                                    
                                    # Define the financial metrics we want to extract
                                    income_statement_metrics = {
                                        'Revenues': 'us-gaap:Revenues',
                                        'CostOfRevenue': 'us-gaap:CostOfRevenue',
                                        'GrossProfit': 'us-gaap:GrossProfit',
                                        'OperatingExpenses': 'us-gaap:OperatingExpenses',
                                        'OperatingIncomeLoss': 'us-gaap:OperatingIncomeLoss',
                                        'NetIncomeLoss': 'us-gaap:NetIncomeLoss',
                                        'EarningsPerShareBasic': 'us-gaap:EarningsPerShareBasic',
                                        'EarningsPerShareDiluted': 'us-gaap:EarningsPerShareDiluted'
                                    }
                                    
                                    balance_sheet_metrics = {
                                        'CashAndCashEquivalents': 'us-gaap:CashAndCashEquivalentsAtCarryingValue',
                                        'TotalAssets': 'us-gaap:Assets',
                                        'TotalCurrentAssets': 'us-gaap:AssetsCurrent',
                                        'TotalLiabilities': 'us-gaap:Liabilities',
                                        'TotalCurrentLiabilities': 'us-gaap:LiabilitiesCurrent',
                                        'TotalStockholdersEquity': 'us-gaap:StockholdersEquity',
                                        'TotalDebt': 'us-gaap:LongTermDebtNoncurrent'
                                    }
                                    
                                    cash_flow_metrics = {
                                        'NetCashProvidedByUsedInOperatingActivities': 'us-gaap:NetCashProvidedByUsedInOperatingActivities',
                                        'NetCashProvidedByUsedInInvestingActivities': 'us-gaap:NetCashProvidedByUsedInInvestingActivities',
                                        'NetCashProvidedByUsedInFinancingActivities': 'us-gaap:NetCashProvidedByUsedInFinancingActivities',
                                        'CapitalExpenditures': 'us-gaap:PaymentsToAcquirePropertyPlantAndEquipment',
                                        'DividendsPaid': 'us-gaap:PaymentsOfDividends'
                                    }
                                    
                                    # Extract facts from the XBRL model
                                    for fact in model_xbrl.facts:
                                        if fact.concept is not None:
                                            concept_name = fact.concept.qname
                                            value = fact.value
                                            context = fact.context
                                            
                                            if context is not None:
                                                period = context.period
                                                if period is not None:
                                                    end_date = period.endDate
                                                    if end_date:
                                                        # Store the fact with its period
                                                        if concept_name not in facts_from_xml:
                                                            facts_from_xml[concept_name] = []
                                                        facts_from_xml[concept_name].append({
                                                            'value': value,
                                                            'end_date': end_date,
                                                            'context': context
                                                        })
                                    
                                    print(f"Extracted {len(facts_from_xml)} unique concepts from XBRL")
                                    
                                    # Map the extracted facts to financial statements
                                    all_metrics = {**income_statement_metrics, **balance_sheet_metrics, **cash_flow_metrics}
                                    
                                    for period in ['annual', 'quarterly']:
                                        for statement, metrics in [('income_statement', income_statement_metrics),
                                                                 ('balance_sheet', balance_sheet_metrics),
                                                                 ('cash_flow', cash_flow_metrics)]:
                                            
                                            # Create a facts dictionary in the format expected by _extract_financial_data
                                            formatted_facts = {}
                                            for metric_name, sec_tag in metrics.items():
                                                if sec_tag in facts_from_xml:
                                                    formatted_facts[sec_tag] = {
                                                        'units': {
                                                            'USD': facts_from_xml[sec_tag]
                                                        }
                                                    }
                                            
                                            # Use the existing extraction logic
                                            df = self._extract_financial_data(formatted_facts, metrics, period)
                                            financial_model[f'{period}_{statement}'] = df
                                    
                                    print("Successfully mapped XBRL facts to financial model")
                                    break
                                    
                                except Exception as e:
                                    print(f"Error parsing XBRL instance document: {e}")
                                    import traceback
                                    traceback.print_exc()
                        else:
                            print(f"Could not access filing directory: {filing_dir_url}")
                        # Only try the first available filing
                        break
                except Exception as e:
                    print(f"Error in XBRL XML fallback: {e}")
                return financial_model
            
        except Exception as e:
            print(f"Error in create_financial_model: {str(e)}")
            return {
                'annual_income_statement': pd.DataFrame(),
                'annual_balance_sheet': pd.DataFrame(),
                'annual_cash_flow': pd.DataFrame(),
                'quarterly_income_statement': pd.DataFrame(),
                'quarterly_balance_sheet': pd.DataFrame(),
                'quarterly_cash_flow': pd.DataFrame()
            }
    
    def _extract_financial_data(self, facts: Dict, metrics: Dict, period: str) -> pd.DataFrame:
        """
        Extract financial data from SEC facts for a given period and set of metrics.
        
        Args:
            facts (Dict): SEC facts data
            metrics (Dict): Dictionary of metric names and their SEC tags
            period (str): 'annual' or 'quarterly'
            
        Returns:
            pd.DataFrame: DataFrame with financial data
        """
        data = {}
        
        for metric_name, sec_tag in metrics.items():
            if sec_tag in facts:
                metric_data = facts[sec_tag]
                units = metric_data.get('units', {})
                
                # Find the appropriate unit (USD is most common)
                unit_key = None
                for key in units.keys():
                    if 'USD' in key or key == 'USD':
                        unit_key = key
                        break
                
                if unit_key:
                    periods = units[unit_key]
                    
                    # Filter by period
                    filtered_periods = []
                    for period_data in periods:
                        if period == 'annual' and period_data.get('form') in ['10-K', '10-K/A']:
                            filtered_periods.append(period_data)
                        elif period == 'quarterly' and period_data.get('form') in ['10-Q', '10-Q/A']:
                            filtered_periods.append(period_data)
                    
                    # Sort by end date
                    filtered_periods.sort(key=lambda x: x.get('end', ''), reverse=True)
                    
                    # Take the most recent 10 periods
                    recent_periods = filtered_periods[:10]
                    
                    for period_data in recent_periods:
                        end_date = period_data.get('end', '')
                        value = period_data.get('val', 0)
                        
                        if end_date not in data:
                            data[end_date] = {}
                        
                        data[end_date][metric_name] = value
        
        # Convert to DataFrame
        if data:
            df = pd.DataFrame.from_dict(data, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            return df
        else:
            return pd.DataFrame()
    
    def create_sensitivity_model(self, financial_model: Dict[str, pd.DataFrame], 
                               ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Create a sensitivity analysis model with operating leverage impacts.
        
        Args:
            financial_model (Dict[str, pd.DataFrame]): Base financial model
            ticker (str): Stock ticker symbol
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing sensitivity analysis dataframes
        """
        try:
            sensitivity_model = {
                'case_summary': pd.DataFrame(),
                'financial_model': pd.DataFrame(),
                'kpi_summary': pd.DataFrame()
            }
            
            # Get the most recent annual income statement for base calculations
            annual_income = financial_model.get('annual_income_statement', pd.DataFrame())
            
            if annual_income.empty:
                print("No annual income statement data available for sensitivity analysis")
                return sensitivity_model
            
            # Create case summary sheet with operating leverage scenarios
            case_summary = self._create_case_summary(annual_income)
            sensitivity_model['case_summary'] = case_summary
            
            # Create enhanced financial model with historical and forecasted data
            enhanced_model = self._create_enhanced_financial_model(financial_model)
            sensitivity_model['financial_model'] = enhanced_model
            
            # Create KPI summary sheet
            kpi_summary = self._create_kpi_summary(financial_model)
            sensitivity_model['kpi_summary'] = kpi_summary
            
            return sensitivity_model
            
        except Exception as e:
            print(f"Error in create_sensitivity_model: {str(e)}")
            return {
                'case_summary': pd.DataFrame(),
                'financial_model': pd.DataFrame(),
                'kpi_summary': pd.DataFrame()
            }
    
    def _create_case_summary(self, annual_income: pd.DataFrame) -> pd.DataFrame:
        """
        Create case summary sheet showing operating leverage impacts.
        
        Args:
            annual_income (pd.DataFrame): Annual income statement data
            
        Returns:
            pd.DataFrame: Case summary with sensitivity scenarios
        """
        if annual_income.empty:
            return pd.DataFrame()
        
        # Get the most recent year's data
        latest_year = annual_income.index[-1]
        base_data = annual_income.loc[latest_year]
        
        # Define sensitivity scenarios
        scenarios = {
            'Base Case': 1.0,
            'Optimistic (+20%)': 1.2,
            'Pessimistic (-20%)': 0.8,
            'High Growth (+50%)': 1.5,
            'Recession (-30%)': 0.7
        }
        
        case_summary_data = []
        
        for scenario_name, revenue_multiplier in scenarios.items():
            # Calculate revenue impact
            base_revenue = base_data.get('Revenues', 0)
            new_revenue = base_revenue * revenue_multiplier
            
            # Calculate operating leverage impact
            # Assume fixed costs remain constant and variable costs scale with revenue
            base_cogs = base_data.get('CostOfRevenue', 0)
            base_opex = base_data.get('OperatingExpenses', 0)
            
            # Assume 70% of COGS is variable, 30% of OpEx is variable
            variable_cogs_ratio = 0.7
            variable_opex_ratio = 0.3
            
            new_cogs = (base_cogs * variable_cogs_ratio * revenue_multiplier + 
                       base_cogs * (1 - variable_cogs_ratio))
            new_opex = (base_opex * variable_opex_ratio * revenue_multiplier + 
                       base_opex * (1 - variable_opex_ratio))
            
            new_gross_profit = new_revenue - new_cogs
            new_operating_income = new_gross_profit - new_opex
            
            # Calculate operating leverage
            revenue_change = (new_revenue - base_revenue) / base_revenue if base_revenue != 0 else 0
            operating_income_change = (new_operating_income - base_data.get('OperatingIncomeLoss', 0)) / base_data.get('OperatingIncomeLoss', 0) if base_data.get('OperatingIncomeLoss', 0) != 0 else 0
            
            operating_leverage = operating_income_change / revenue_change if revenue_change != 0 else 0
            
            case_summary_data.append({
                'Scenario': scenario_name,
                'Revenue': new_revenue,
                'COGS': new_cogs,
                'Gross Profit': new_gross_profit,
                'Operating Expenses': new_opex,
                'Operating Income': new_operating_income,
                'Revenue Change %': revenue_change * 100,
                'Operating Income Change %': operating_income_change * 100,
                'Operating Leverage': operating_leverage
            })
        
        return pd.DataFrame(case_summary_data)
    
    def _create_enhanced_financial_model(self, financial_model: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create enhanced financial model with historical and forecasted data.
        
        Args:
            financial_model (Dict[str, pd.DataFrame]): Base financial model
            
        Returns:
            pd.DataFrame: Enhanced financial model
        """
        # Combine all financial data into one comprehensive model
        enhanced_data = {}
        
        # Add historical data
        for statement_type in ['income_statement', 'balance_sheet', 'cash_flow']:
            for period in ['annual', 'quarterly']:
                key = f'{period}_{statement_type}'
                if key in financial_model and not financial_model[key].empty:
                    df = financial_model[key]
                    for date in df.index:
                        for column in df.columns:
                            enhanced_data[f"{period}_{statement_type}_{column}"] = {
                                'date': date,
                                'value': df.loc[date, column],
                                'type': 'historical',
                                'period': period,
                                'statement': statement_type
                            }
        
        # Add forecasted data (simple linear projection for demonstration)
        if enhanced_data:
            # Get the most recent dates for each period
            annual_dates = []
            quarterly_dates = []
            
            for key, data in enhanced_data.items():
                if data['period'] == 'annual':
                    annual_dates.append(data['date'])
                elif data['period'] == 'quarterly':
                    quarterly_dates.append(data['date'])
            
            if annual_dates:
                latest_annual = max(annual_dates)
                # Add 3 years of forecast
                for i in range(1, 4):
                    forecast_date = latest_annual + pd.DateOffset(years=i)
                    enhanced_data[f"forecast_annual_year_{i}"] = {
                        'date': forecast_date,
                        'value': 0,  # Placeholder for forecasted values
                        'type': 'forecast',
                        'period': 'annual',
                        'statement': 'projection'
                    }
            
            if quarterly_dates:
                latest_quarterly = max(quarterly_dates)
                # Add 4 quarters of forecast
                for i in range(1, 5):
                    forecast_date = latest_quarterly + pd.DateOffset(months=i*3)
                    enhanced_data[f"forecast_quarterly_q{i}"] = {
                        'date': forecast_date,
                        'value': 0,  # Placeholder for forecasted values
                        'type': 'forecast',
                        'period': 'quarterly',
                        'statement': 'projection'
                    }
        
        # Convert to DataFrame
        if enhanced_data:
            df = pd.DataFrame.from_dict(enhanced_data, orient='index')
            df = df.sort_values('date')
            return df
        else:
            return pd.DataFrame()
    
    def _create_kpi_summary(self, financial_model: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create KPI summary sheet with key financial metrics.
        
        Args:
            financial_model (Dict[str, pd.DataFrame]): Base financial model
            
        Returns:
            pd.DataFrame: KPI summary sheet
        """
        kpi_data = []
        
        # Calculate KPIs from available data
        annual_income = financial_model.get('annual_income_statement', pd.DataFrame())
        annual_balance = financial_model.get('annual_balance_sheet', pd.DataFrame())
        annual_cash_flow = financial_model.get('annual_cash_flow', pd.DataFrame())
        
        if not annual_income.empty:
            latest_income = annual_income.iloc[-1]
            
            # Profitability KPIs
            revenue = latest_income.get('Revenues', 0)
            net_income = latest_income.get('NetIncomeLoss', 0)
            operating_income = latest_income.get('OperatingIncomeLoss', 0)
            gross_profit = latest_income.get('GrossProfit', 0)
            
            if revenue != 0:
                kpi_data.extend([
                    {'KPI': 'Net Profit Margin', 'Value': (net_income / revenue) * 100, 'Unit': '%'},
                    {'KPI': 'Operating Margin', 'Value': (operating_income / revenue) * 100, 'Unit': '%'},
                    {'KPI': 'Gross Margin', 'Value': (gross_profit / revenue) * 100, 'Unit': '%'}
                ])
        
        if not annual_balance.empty:
            latest_balance = annual_balance.iloc[-1]
            
            # Efficiency KPIs
            total_assets = latest_balance.get('TotalAssets', 0)
            total_equity = latest_balance.get('TotalStockholdersEquity', 0)
            
            if total_assets != 0:
                kpi_data.append({'KPI': 'Return on Assets (ROA)', 'Value': (net_income / total_assets) * 100, 'Unit': '%'})
            
            if total_equity != 0:
                kpi_data.append({'KPI': 'Return on Equity (ROE)', 'Value': (net_income / total_equity) * 100, 'Unit': '%'})
            
            # Liquidity KPIs
            current_assets = latest_balance.get('TotalCurrentAssets', 0)
            current_liabilities = latest_balance.get('TotalCurrentLiabilities', 0)
            
            if current_liabilities != 0:
                kpi_data.append({'KPI': 'Current Ratio', 'Value': current_assets / current_liabilities, 'Unit': 'x'})
        
        if not annual_cash_flow.empty:
            latest_cash_flow = annual_cash_flow.iloc[-1]
            
            # Cash Flow KPIs
            operating_cash_flow = latest_cash_flow.get('NetCashProvidedByUsedInOperatingActivities', 0)
            capital_expenditures = abs(latest_cash_flow.get('CapitalExpenditures', 0))
            
            if capital_expenditures != 0:
                kpi_data.append({'KPI': 'Operating Cash Flow to CapEx', 'Value': operating_cash_flow / capital_expenditures, 'Unit': 'x'})
        
        # Add growth metrics if multiple years available
        if len(annual_income) >= 2:
            current_revenue = annual_income.iloc[-1].get('Revenues', 0)
            previous_revenue = annual_income.iloc[-2].get('Revenues', 0)
            
            if previous_revenue != 0:
                revenue_growth = ((current_revenue - previous_revenue) / previous_revenue) * 100
                kpi_data.append({'KPI': 'Revenue Growth (YoY)', 'Value': revenue_growth, 'Unit': '%'})
        
        return pd.DataFrame(kpi_data)
    
    def get_filing_content(self, accession_number: str, primary_document: str) -> str:
        """
        Get the content of a specific SEC filing.
        
        Args:
            accession_number (str): SEC accession number
            primary_document (str): Primary document name
            
        Returns:
            str: Filing content
        """
        try:
            # Format accession number
            accession_number = accession_number.replace('-', '')
            
            # Construct URL for filing
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{accession_number}/{primary_document}"
            
            response = self.session.get(filing_url)
            
            if response.status_code == 200:
                return response.text
            else:
                print(f"Error fetching filing content: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"Error in get_filing_content: {str(e)}")
            return ""
    
    def export_to_excel(self, financial_model: Dict[str, pd.DataFrame], 
                       sensitivity_model: Dict[str, pd.DataFrame], 
                       ticker: str, filename: str = None) -> str:
        """
        Export financial models to Excel file with multiple sheets.
        
        Args:
            financial_model (Dict[str, pd.DataFrame]): Base financial model
            sensitivity_model (Dict[str, pd.DataFrame]): Sensitivity analysis model
            ticker (str): Stock ticker symbol
            filename (str): Output filename (optional)
            
        Returns:
            str: Path to the exported Excel file
        """
        # Ensure the Storage directory exists
        storage_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Storage')
        os.makedirs(storage_dir, exist_ok=True)
        
        if filename is None:
            filename = f"{ticker}_financial_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        filepath = os.path.join(storage_dir, filename)
        
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Write base financial model sheets
                for sheet_name, df in financial_model.items():
                    if not df.empty:
                        df.to_excel(writer, sheet_name=sheet_name.replace('_', ' ').title())
                
                # Write sensitivity model sheets
                for sheet_name, df in sensitivity_model.items():
                    if not df.empty:
                        df.to_excel(writer, sheet_name=f"{sheet_name.replace('_', ' ').title()}")
                
                # Write summary sheet
                summary_data = {
                    'Metric': ['Ticker', 'Model Created', 'Data Points', 'Scenarios Analyzed'],
                    'Value': [
                        ticker,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        sum(len(df) for df in financial_model.values() if not df.empty),
                        len(sensitivity_model.get('case_summary', pd.DataFrame()))
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            print(f"Financial model exported to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error exporting to Excel: {e}")
            return ""

# Example usage and testing
if __name__ == "__main__":
    # Initialize the SEC File Sourcer
    sourcer = SECFileSourcer()
    
    # Example ticker - testing with MSFT instead of AAPL
    ticker = "MSFT"
    
    print(f"Finding SEC filings for {ticker}...")
    filings = sourcer.find_sec_filings(ticker)
    print(f"Found {len(filings)} filings")
    
    if not filings.empty:
        print("\nRecent filings:")
        print(filings.head())
    
    print(f"\nCreating financial model for {ticker}...")
    financial_model = sourcer.create_financial_model(ticker)
    
    # Only proceed if at least one financial model dataframe is non-empty
    if any(not df.empty for df in financial_model.values()):
        print(f"\nCreating sensitivity analysis for {ticker}...")
        sensitivity_model = sourcer.create_sensitivity_model(financial_model, ticker)
        
        # Export to Excel
        excel_file = sourcer.export_to_excel(financial_model, sensitivity_model, ticker)
        print(f"\nModel creation complete! Check the Excel file: {excel_file}")
    else:
        print("\nNo financial data was successfully pulled. Excel file will not be created.") 