// src/components/TooltipContext.js
import React, { createContext, useState } from 'react';

export const TooltipContext = createContext();

export const TooltipProvider = ({ children }) => {
    const [tooltipData, setTooltipData] = useState(null);

    return (
        <TooltipContext.Provider value={{ tooltipData, setTooltipData }}>
            {children}
        </TooltipContext.Provider>
    );
};