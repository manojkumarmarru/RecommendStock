import React, { useState } from 'react';
import axios from 'axios';
import StockChart from './StockChart';
import { TooltipProvider } from './TooltipContext';
import { Line } from 'react-chartjs-2'; // Import the Line component from react-chartjs-2
import 'chart.js/auto';
import '../Styles/StockComparison.css';

const apiKey = 'jZ3KwctIb3G2e8zK4OTShjr5UpW3S53G';

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
    const response = await axios.get(`http://localhost:5143/api/StockPrediction/${symbol}`);
    return response.data;
};

const ComparisonChart = ({ comparisonData }) => {
    console.log('ComparisonChart component rendered'); // Debugging log
    console.log('ComparisonChart received data:', comparisonData); // Debugging log

    const data = {
        labels: comparisonData.map(entry => entry.date),
        datasets: [
            {
                label: 'Actual Close',
                data: comparisonData.map(entry => entry.actualClose),
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                fill: false,
            },
            {
                label: 'Predicted Close',
                data: comparisonData.map(entry => entry.predictedClose),
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                fill: false,
            },
        ],
    };

    const options = {
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
    };

    return <Line data={data} options={options} />;
};

const StockComparison = () => {
    const [symbol, setSymbol] = useState('');
    const [priceData, setPriceData] = useState([]);
    const [emaData, setEmaData] = useState([]);
    const [rsiData, setRsiData] = useState([]);
    const [ratingData, setRatingData] = useState([]);
    const [prediction, setPrediction] = useState(null);
    const [comparisonData, setComparisonData] = useState(null); // Initialize comparisonData with null

    const fetchData = async () => {
        if (!symbol) {
            alert('Please enter a stock symbol');
            return;
        }

        try {
            const priceData = await fetchStock(symbol);
            // const emaData = await fetchIndicatorData(symbol, 'ema');
            // const rsiData = await fetchIndicatorData(symbol, 'rsi');
            // const ratingData = await fetchHistoricalRating(symbol);
            const prediction = await fetchPrediction(symbol);

            console.log('Fetched Price Data:', priceData); // Debugging step
            console.log('Fetched Prediction Data:', prediction); // Debugging step

            if (prediction) {
                console.log('Prediction Data:', prediction);
                console.log('Prediction Properties:', Object.keys(prediction)); // Log the properties of prediction
                console.log('Comparison Data:', prediction.comparison); // Log the comparison property
            }

            setPriceData(priceData);
            setEmaData(emaData);
            setRsiData(rsiData);
            setRatingData(ratingData);
            setPrediction(prediction);
            setComparisonData(prediction?.forecast || []); // Ensure comparisonData is an array
        } catch (error) {
            console.error('Error fetching data:', error);
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
                    <button onClick={fetchData}>Get Data</button>
                </div>
                {prediction && (
                <div className="prediction">
                    <h3>Recommendation: {prediction.recommendation}</h3>
                </div>
                )}
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
                                            <th>Volume</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {priceData.map((entry, index) => (
                                            <tr key={index}>
                                                <td>{entry.date}</td>
                                                <td>{entry.open}</td>
                                                <td>{entry.high}</td>
                                                <td>{entry.low}</td>
                                                <td>{entry.close}</td>
                                                <td>{entry.volume}</td>
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
                            <StockChart canvasId="priceChart" datasets={[
                                {
                                    label: 'Stock Price',
                                    data: priceData.map(entry => ({ x: entry.date, y: entry.close })),
                                    borderColor: 'rgba(75, 192, 192, 1)',
                                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                                    fill: false,
                                },
                                {
                                    label: 'Predicted Close',
                                    data: prediction?.forecast.map(entry => ({ x: entry.date, y: entry.forecastedClose })) || [],
                                    borderColor: 'rgba(255, 99, 132, 1)',
                                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                                    fill: false,
                                }
                            ]} className="chart" />
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
                                                    <td>{entry.date}</td>
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
            </div>
        </TooltipProvider>
    );
};

export default StockComparison;