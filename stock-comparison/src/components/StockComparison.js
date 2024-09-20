import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Chart, registerables } from 'chart.js'; // Import Chart and registerables from chart.js
import 'chartjs-adapter-date-fns'; // Import the date adapter
import 'chart.js/auto';
import '../Styles/StockComparison.css';

Chart.register(...registerables); // Register the necessary components

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

const fetchPrediction = async (symbol) => {
    const response = await axios.get(`http://localhost:5143/api/StockPrediction/${symbol}`);
    return response.data;
};

const StockComparison = () => {
    const [symbol, setSymbol] = useState('');
    const [priceData, setPriceData] = useState([]);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const chartRef = useRef(null);

    const fetchData = async () => {
        if (!symbol) {
            alert('Please enter a stock symbol');
            return;
        }

        setPriceData([]);
        setPrediction(null);
        setLoading(true);
        try {
            const priceData = await fetchStock(symbol);
            const prediction = await fetchPrediction(symbol);
            setPriceData(priceData);
            setPrediction(prediction);
            console.log('Prediction:', prediction);
        } catch (error) {
            console.error('Error fetching data:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (chartRef.current) {
            chartRef.current.destroy();
        }

        if (priceData.length > 0 && prediction) {
            const ctx = document.getElementById('priceChart').getContext('2d');
            chartRef.current = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: priceData.map(entry => entry.date),
                    datasets: [
                        {
                            label: 'Actual Close',
                            data: priceData.map(entry => entry.close),
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            fill: false,
                        },
                        {
                            label: 'Predicted Close',
                            data: prediction.forecast.map(entry => entry.forecastedClose),
                            borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            fill: false,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            type: 'time',
                            time: {
                                unit: 'day',
                            },
                        },
                        y: {
                            beginAtZero: false,
                        },
                    },
                },
            });
        }
    }, [priceData, prediction]);

    return (
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
                <div className="loading">Loading...</div>
            ) : (
                <div className="main-content">
                    <div className="side-content">
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
                                        {priceData.map((entry, index) => (
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
                    <div className="center-content">
                        <div className="chart-container">
                            <h4>Stock Price</h4>
                            <canvas id="priceChart"></canvas>
                        </div>
                    </div>
                    <div className="side-content">
                        <div className="table-container scrollable-table">
                            <h4>Stock Prediction Table</h4>
                            {prediction && prediction.forecast && prediction.forecast.length > 0 ? (
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Date</th>
                                            <th>Forecasted Close</th>
                                            <th>Lower Bound Close</th>
                                            <th>Upper Bound Close</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {prediction.forecast
                                            .filter(entry => {
                                                const entryDate = new Date(entry.date);
                                                const today = new Date();
                                                today.setHours(0, 0, 0, 0);
                                                return entryDate >= today;
                                            })
                                            .sort((a, b) => new Date(a.date) - new Date(b.date))
                                            .map((entry, index) => (
                                                <tr key={index}>
                                                    <td>{new Date(entry.date).toLocaleDateString()}</td>
                                                    <td>{entry.forecastedClose}</td>
                                                    <td>{entry.lowerBoundClose}</td>
                                                    <td>{entry.upperBoundClose}</td>
                                                </tr>
                                            ))}
                                    </tbody>
                                </table>
                            ) : (
                                <p>No data available</p>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default StockComparison