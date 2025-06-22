import http from 'k6/http'
import { sleep, check } from 'k6'
import { Rate, Trend, Counter, Gauge } from 'k6/metrics'

let errorRate = new Rate('errors');
let responseTime = new Trend('response_time');
let requestCount = new Counter('requests_total');
let concurrentUsers = new Gauge('concurrent_users');
let successfulRequests = new Counter('successful_requests');
let failedRequests = new Counter('failed_requests');

export const options = {
    stages: [
        { duration: '2m', target: 5 },   
        { duration: '3m', target: 7 },  
        
        { duration: '1m', target: 15 },  
        { duration: '2m', target: 15 },  
        { duration: '1m', target: 10 },  
        { duration: '2m', target: 10 },  
        
        { duration: '1m', target: 25 },  
        { duration: '3m', target: 20 },  
        { duration: '1m', target: 15 },  
        { duration: '2m', target: 15 },  

        { duration: '1m', target: 10 },  
        { duration: '2m', target: 5 },  
        { duration: '2m', target: 5 },  
        { duration: '2m', target: 2 },   

        { duration: '1m', target: 20 },  
        { duration: '2m', target: 20 },  
        { duration: '30s', target: 30 }, 
        { duration: '1m', target: 30 },  

        { duration: '2m', target: 5 },  
        { duration: '2m', target: 5 },  
        { duration: '2m', target: 2 },   

        
        { duration: '1m', target: 0 },  
    ],
}

const BASE_URL = 'http://my-app-name-alb-19580245.us-east-2.elb.amazonaws.com'

const STOCK_TICKERS = [
  'AAPL',  
  'GOOGL',  
  'GOOG',   
  'MSFT',   
  'AMZN',   
  'TSLA',   
  'META',   
  'NVDA',   
  'NFLX',   
  'AMD',    
  'CRM',    
  'ORCL',   
  'IBM',    
  'UBER',   
  'LYFT',   
  'SNAP',   
  'PYPL',   
  'SQ',     
  'SHOP',   
  'ADBE',   
  'ZOOM',   
  'SPOT',   
  'TWLO',   
  'V',      
  'MA',     
  'DIS',    
  'BABA',   
];

const FORECAST_DAYS = [5, 10, 15, 20, 30];

export function setup() {
    console.log('🚀 Starting ECS Elasticity Test...');
            console.log('🏔️ MULTI-PEAK ELASTICITY TEST: Tests multiple scaling events');
        console.log('📊 This test creates 3 distinct load peaks to test repeated scaling behavior');

    let response = http.get(`${BASE_URL}/`);
    
    if (response.status !== 200) {
        console.error(`API not accessible. Status: ${response.status}`);
        return null;
    }
  
    console.log('API is accessible. Beginning elasticity test...');
    console.log('Watch your ECS metrics for task scaling events!');
    
    return { 
        baseUrl: BASE_URL,
        testStartTime: Date.now()
    };
}

export default function(data) {
    if (!data) {
        console.error('Setup failed');
        return;
    }

    concurrentUsers.add(__VU);

    const requestType = Math.random();
    
    if (requestType < 0.7) {
        performStockForecastRequest(data);
    }
    else {
        performBurstRequests(data);
    }

    const sleepTime = Math.random() * 2 + 0.5; 
    sleep(sleepTime);
}

function performStockForecastRequest(data) {
    const stockTicker = STOCK_TICKERS[Math.floor(Math.random() * STOCK_TICKERS.length)];
    const forecastDays = FORECAST_DAYS[Math.floor(Math.random() * FORECAST_DAYS.length)];

    const payload = JSON.stringify({
        days: forecastDays
    });

    const params = {
        headers: {
            'Content-Type': 'application/json',
            'User-Agent': `k6-elasticity-test-vu-${__VU}`,
        },
        timeout: '45s', 
    };

    const startTime = Date.now();
    let response = http.post(`${data.baseUrl}/forecast/${stockTicker}`, payload, params);
    const endTime = Date.now();
    
    const duration = endTime - startTime;
    
    requestCount.add(1);
    responseTime.add(duration);

    const isSuccess = check(response, {
        'status is 200': (r) => r.status === 200,
        'response has stock_ticker': (r) => {
            try {
                const body = JSON.parse(r.body);
                return body.stock_ticker === stockTicker;
            } catch (e) {
                return false;
            }
        },
        'response has forecast_prices': (r) => {
            try {
                const body = JSON.parse(r.body);
                return body.forecast_prices && typeof body.forecast_prices === 'object';
            } catch (e) {
                return false;
            }
        },
        'forecast_prices has correct number of days': (r) => {
            try {
                const body = JSON.parse(r.body);
                const forecastDates = Object.keys(body.forecast_prices);
                return forecastDates.length === forecastDays;
            } catch (e) {
                return false;
            }
        },
        'response time acceptable': (r) => r.timings.duration < 45000,
    });

    if (isSuccess) {
        successfulRequests.add(1);
        errorRate.add(0);
        
        if (duration > 20000) {
            console.log(`Slow response detected: ${stockTicker} took ${duration}ms (possible scaling event)`);
        } else if (duration < 2000) {
            console.log(`Fast response: ${stockTicker} completed in ${duration}ms (scaled up)`);
        }
    } else {
        failedRequests.add(1);
        errorRate.add(1);
        console.error(`Request failed for ${stockTicker}: Status ${response.status}, Duration: ${duration}ms`);
        
        if (response.status === 503 || response.status === 502) {
            console.log(`🔄 Service unavailable - likely scaling in progress`);
        }
    }
}

function performBurstRequests(data) {
    const burstCount = Math.floor(Math.random() * 3) + 2; 
    
    for (let i = 0; i < burstCount; i++) {
        const stockTicker = STOCK_TICKERS[Math.floor(Math.random() * STOCK_TICKERS.length)];
        const payload = JSON.stringify({ days: 5 }); 
        
        const params = {
            headers: { 'Content-Type': 'application/json' },
            timeout: '30s',
        };
        
        const response = http.post(`${data.baseUrl}/forecast/${stockTicker}`, payload, params);
        
        requestCount.add(1);
        
        if (response.status === 200) {
            successfulRequests.add(1);
            errorRate.add(0);
        } else {
            failedRequests.add(1);
            errorRate.add(1);
        }
        
        sleep(0.1); 
    }
}

export function teardown(data) {
    if (data) {
        const testDuration = (Date.now() - data.testStartTime) / 1000;
        console.log('\n ECS Elasticity Test Completed!');
        console.log(`Total test duration: ${Math.round(testDuration)}s`);
    }
}