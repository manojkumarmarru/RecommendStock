// src/App.js
import React from 'react';
import StockComparisonEnhanced from './components/StockComparisonEnhanced';
import './App.css';
import StockComparison from './components/StockComparison';

function App() {
    return (
        <div className="App">
            {/* <StockComparison /> */}
            <StockComparisonEnhanced />
        </div>
    );
}

export default App;