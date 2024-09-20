import React, { useState } from 'react';
import axios from 'axios';
import StockChart from './StockChart';
import { TooltipProvider } from './TooltipContext';
import 'chart.js/auto';
import '../Styles/StockComparison.css';

const apiKey = 'fdBhj6FJhbAKaYyTh5fU3pwUvPY5X32E';

const fetchStock = async (symbol) => {
    const response = await axios.get(`https://financialmodelingprep.com/api/v3/historical-price-full/${symbol}?apikey=${apiKey}`);
    const data = response.data.historical;
    return data.map(entry => ({
        date: entry.date,
        open: entry.open,
        high: entry.high,
        low: entry.low,
        close: entry.close,
        volume: entry.volume
    }));
};

const fetchLSTMPrediction = async (symbol) => {
    const response = await axios.get(`http://localhost:5143/api/LSTMStockPrediction/${symbol}`);
    return response.data;
};

const fetchMLPrediction = async (symbol) => {
    const response = await axios.get(`http://localhost:5143/api/StockPrediction/${symbol}`);
    return response.data;
};

const StockComparisonEnhanced = () => {
    const [symbol, setSymbol] = useState('');
    const [priceData, setPriceData] = useState([]);
    const [lstmpredictionData, setLSTMPredictionData] = useState(null);
    const [mlpredictionData, setMLPredictionData] = useState(null);
    const [loading, setLoading] = useState(false);

    const fetchData = async () => {
        if (!symbol) {
            alert('Please enter a stock symbol');
            return;
        }

        setPriceData([]);
        setLSTMPredictionData(null);
        setMLPredictionData(null);
        setLoading(true);
        try {
            const priceData = await fetchStock(symbol);
            const lstmpredictionData = await fetchLSTMPrediction(symbol);
            const mlpredictionData = await fetchMLPrediction(symbol);
            setPriceData(priceData);
            setLSTMPredictionData(lstmpredictionData);
            setMLPredictionData(mlpredictionData);
            console.log('LSTM Prediction:', lstmpredictionData);
            console.log('ML Prediction:', mlpredictionData);
        } catch (error) {
            console.error('Error fetching data:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <TooltipProvider>
            <div className="container">
                <h1>Stock Forecasting</h1>
                <div className="input-container">
                    <input
                        type="text"
                        value={symbol}
                        onChange={(e) => setSymbol(e.target.value)}
                        placeholder="Enter Stock Symbol"
                    />
                    <button onClick={fetchData}>Submit</button>
                </div>
                {loading ? (
                    <div className="loading mt-10">Loading...</div>
                ) : (
                    <>
                        <div className="main-content">
                            <div className="part-content">
                                <div className="table-container scrollable-table">
                                    <h4>Stock Data Table</h4>
                                    {priceData.length > 0 ? (
                                        <table>
                                            <thead>
                                                <tr>
                                                    <th>Date</th>
                                                    <th>Open</th>
                                                    <th>High</th>
                                                    <th>Low</th>
                                                    <th>Close</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {priceData.sort((a, b) => new Date(b.date) - new Date(a.date)).map((entry, index) => (
                                                    <tr key={index}>
                                                        <td>{new Date(entry.date).toLocaleDateString()}</td>
                                                        <td>{entry.open}</td>
                                                        <td>{entry.high}</td>
                                                        <td>{entry.low}</td>
                                                        <td>{entry.close}</td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    ) : (
                                        <p>No data available</p>
                                    )}
                                </div>
                            </div>
                            <div className='part-content'>
                                <div className="chart-container">
                                    <StockChart canvasId="priceChart" datasets={[
                                        {
                                            label: 'Historical Price',
                                            data: priceData.map(entry => ({ x: entry.date, y: entry.close })),
                                            borderColor: 'rgba(75, 192, 192, 1)',
                                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                                            fill: false,
                                        },
                                        {
                                            label: 'LSTM Predicted Close',
                                            data: lstmpredictionData?.forecastedAdjClose.map((price, index) => ({
                                                x: new Date('2024-03-22').setDate(new Date('2024-03-22').getDate() + index),
                                                y: price
                                            })) || [],
                                            borderColor: 'rgba(255, 99, 132, 1)',
                                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                                            fill: false,
                                        },
                                        {
                                            label: 'Mcirosoft ML TImeSeries Predicted Close',
                                            data: mlpredictionData?.forecast.map((entry) => ({
                                                x: entry.date,
                                                y: entry.forecastedClose
                                            })) || [],
                                            borderColor: 'rgba(54, 162, 235, 1)',
                                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                                            fill: false,
                                        }
                                    ]} className="chart" />
                                </div>
                            </div>
                            <div className="part-content">
                                <div className="table-container scrollable-table">
                                    <h4>Stock Prediction Table</h4>
                                    {lstmpredictionData && mlpredictionData ? (
                                        <table>
                                            <thead>
                                                <tr>
                                                    <th>Date</th>
                                                    <th>Actual Close</th>
                                                    <th>ML Predicted Close</th>
                                                    <th>LSTM Predicted Close</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {lstmpredictionData.forecastedAdjClose.map((price, index) => {
                                                    const date = new Date(new Date('2024-03-22').setDate(new Date('2024-03-22').getDate() + index)).toLocaleDateString();
                                                    const actualClose = priceData.find(entry => new Date(entry.date).toLocaleDateString() === date)?.close || 'N/A';
                                                    return (
                                                        <tr key={index}>
                                                            <td>{date}</td>
                                                            <td>{actualClose}</td>
                                                            <td>{mlpredictionData.forecast[index]?.forecastedClose}</td>
                                                            <td>{price}</td>
                                                        </tr>
                                                    );
                                                })}
                                            </tbody>
                                        </table>
                                    ) : (
                                        <p>No data available</p>
                                    )}
                                </div>
                            </div>
                        </div>
                    </>
                )}
            </div>
        </TooltipProvider>
    );
};

export default StockComparisonEnhanced;