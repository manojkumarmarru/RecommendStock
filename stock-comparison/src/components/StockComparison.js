// src/components/StockComparison.js
import React, { useState } from 'react';
import axios from 'axios';
import StockChart from './StockChart';
import { TooltipProvider } from './TooltipContext';
import '../Styles/StockComparison.css';

const apiKey = 'fdBhj6FJhbAKaYyTh5fU3pwUvPY5X32E';

const fetchStockData = async (symbol) => {
    const response = await axios.get(`https://financialmodelingprep.com/api/v3/historical-price-full/${symbol}?apikey=${apiKey}`);
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

const fetchHistoricalRating = async (symbol) => {
    const response = await axios.get(`https://financialmodelingprep.com/api/v3/historical-rating/${symbol}?apikey=${apiKey}`);
    const data = response.data;
    return data.map(entry => ({
        x: entry.date,
        y: entry.ratingScore,
        recommendation: entry.ratingRecommendation
    }));
};

const fetchPrediction = async (symbol) => {
    const response = await axios.get(`https://curly-invention-q956vrpr5qj24wg5-5143.app.github.dev/api/StockPrediction/${symbol}`);
    return response.data;
};

const StockComparison = () => {
    const [symbol, setSymbol] = useState('');
    const [priceData, setPriceData] = useState([]);
    const [emaData, setEmaData] = useState([]);
    const [rsiData, setRsiData] = useState([]);
    const [ratingData, setRatingData] = useState([]);
    const [prediction, setPrediction] = useState(null);
    
    const fetchData = async () => {
        if (!symbol) {
            alert('Please enter a stock symbol');
            return;
        }

        try {
            const priceData = await fetchStockData(symbol);
            const emaData = await fetchIndicatorData(symbol, 'ema');
            const rsiData = await fetchIndicatorData(symbol, 'rsi');
            const ratingData = await fetchHistoricalRating(symbol);
            const prediction = await fetchPrediction(symbol);

            setPriceData(priceData);
            setEmaData(emaData);
            setRsiData(rsiData);
            setRatingData(ratingData);
            setPrediction(prediction);
        } catch (error) {
            console.error('Error fetching data:', error);
        }
    };

    return (
        <TooltipProvider>
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
                {prediction && (
                <div className="prediction">
                    <h3>Recommendation: {prediction.recommendation}</h3>
                </div>
                )}
                <div className="chart-wrapper">
                    <div className="chart-container">
                        <h4>Stock Price</h4>
                        <StockChart canvasId="priceChart" datasets={[{
                            label: 'Stock Price',
                            data: priceData,
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            fill: false,
                        }]} className="chart" />
                    </div>
                    <div className="chart-container">
                        <h4>EMA</h4>
                        <StockChart canvasId="emaChart" datasets={[{
                            label: 'EMA',
                            data: emaData,
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            fill: false,
                        }]} className="chart" />
                    </div>
                </div>
                <div className="chart-wrapper">
                    <div className="chart-container">
                        <h4>RSI</h4>
                        <StockChart canvasId="rsiChart" datasets={[{
                            label: 'RSI',
                            data: rsiData,
                            borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            fill: false,
                        }]} className="chart" />
                    </div>
                    <div className="chart-container">
                        <h4>Historical Rating</h4>
                        <StockChart canvasId="ratingChart" datasets={[{
                            label: 'Rating Score',
                            data: ratingData,
                            borderColor: 'rgba(255, 206, 86, 1)',
                            backgroundColor: 'rgba(255, 206, 86, 0.2)',
                            fill: false,
                        }]} className="chart" />
                    </div>
                </div>
            </div>
        </TooltipProvider>
    );
};

export default StockComparison;