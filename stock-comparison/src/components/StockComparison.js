// src/components/StockComparison.js
import React, { useState } from 'react';
import axios from 'axios';
import StockChart from './StockChart';

const apiKey = 'jZ3KwctIb3G2e8zK4OTShjr5UpW3S53G';

const fetchStockData = async (symbol) => {
    const response = await axios.get(`https://financialmodelingprep.com/api/v3/historical-price-full/${symbol}?timeseries=1825&apikey=${apiKey}`);
    const data = response.data.historical;
    return data.map(entry => ({
        x: entry.date,
        y: entry.close
    }));
};

const fetchIndicatorData = async (symbol, indicator) => {
    const response = await axios.get(`https://financialmodelingprep.com/api/v3/technical_indicator/1day/${symbol}?type=${indicator}&period=1825&apikey=${apiKey}`);
    const data = response.data;
    return data.map(entry => ({
        x: entry.date,
        y: entry[indicator]
    }));
};

const StockComparison = () => {
    const [symbol, setSymbol] = useState('');
    const [priceData, setPriceData] = useState([]);
    const [smaData, setSmaData] = useState([]);
    const [emaData, setEmaData] = useState([]);
    const [wmaData, setWmaData] = useState([]);

    const fetchData = async () => {
        if (!symbol) {
            alert('Please enter a stock symbol');
            return;
        }

        try {
            const priceData = await fetchStockData(symbol);
            const smaData = await fetchIndicatorData(symbol, 'sma');
            const emaData = await fetchIndicatorData(symbol, 'ema');
            const wmaData = await fetchIndicatorData(symbol, 'wma');

            setPriceData(priceData);
            setSmaData(smaData);
            setEmaData(emaData);
            setWmaData(wmaData);
        } catch (error) {
            console.error('Error fetching data:', error);
        }
    };

    const datasets = [
        {
            label: 'Stock Price',
            data: priceData,
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            fill: false,
        },
        {
            label: 'SMA',
            data: smaData,
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            fill: false,
        },
        {
            label: 'EMA',
            data: emaData,
            borderColor: 'rgba(54, 162, 235, 1)',
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            fill: false,
        },
        {
            label: 'WMA',
            data: wmaData,
            borderColor: 'rgba(255, 206, 86, 1)',
            backgroundColor: 'rgba(255, 206, 86, 0.2)',
            fill: false,
        }
    ];

    return (
        <div className="container">
            <div className="input-container">
                <input
                    type="text"
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value)}
                    placeholder="Enter Stock Symbol"
                />
                <button onClick={fetchData}>Get Data</button>
            </div>
            <div className="chart-wrapper">
                <div className="chart-container">
                    <StockChart canvasId="priceChart" datasets={datasets} className="chart" />
                </div>
            </div>
        </div>
    );
};

export default StockComparison;