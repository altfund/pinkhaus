import { useState, useEffect, createContext, useContext, useCallback } from 'react';
import { web3Provider } from '../utils/web3Provider';

const Web3Context = createContext({});

export function Web3Provider({ children }) {
  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [account, setAccount] = useState(null);
  const [chainId, setChainId] = useState(null);
  const [provider, setProvider] = useState(null);
  const [signer, setSigner] = useState(null);

  // Auto-connect if previously connected
  useEffect(() => {
    if (web3Provider.web3Modal?.cachedProvider) {
      connect();
    }
  }, []);

  const connect = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const connection = await web3Provider.connect();
      
      setProvider(connection.provider);
      setSigner(connection.signer);
      setAccount(connection.address);
      setChainId(connection.chainId);
      setConnected(true);
    } catch (err) {
      setError(err.message);
      console.error('Failed to connect:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const disconnect = useCallback(async () => {
    await web3Provider.disconnect();
    setConnected(false);
    setAccount(null);
    setChainId(null);
    setProvider(null);
    setSigner(null);
  }, []);

  const switchNetwork = useCallback(async (newChainId) => {
    setLoading(true);
    setError(null);
    
    try {
      await web3Provider.switchNetwork(newChainId);
      // Page will reload after network switch
    } catch (err) {
      setError(err.message);
      console.error('Failed to switch network:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const value = {
    connected,
    loading,
    error,
    account,
    chainId,
    provider,
    signer,
    connect,
    disconnect,
    switchNetwork,
    web3Provider,
  };

  return (
    <Web3Context.Provider value={value}>
      {children}
    </Web3Context.Provider>
  );
}

export function useWeb3() {
  const context = useContext(Web3Context);
  
  if (!context) {
    throw new Error('useWeb3 must be used within Web3Provider');
  }
  
  return context;
}

// Custom hooks for specific functionality
export function useOminariContract() {
  const { web3Provider, connected } = useWeb3();
  
  return {
    createSession: useCallback(async (initialBankroll) => {
      if (!connected) throw new Error('Wallet not connected');
      return web3Provider.createSession(initialBankroll);
    }, [connected, web3Provider]),
    
    placeBet: useCallback(async (sessionId, marketId, stake, outcome) => {
      if (!connected) throw new Error('Wallet not connected');
      return web3Provider.placeBet(sessionId, marketId, stake, outcome);
    }, [connected, web3Provider]),
    
    optimizePortfolio: useCallback(async (sessionId, marketIds, chunkDuration) => {
      if (!connected) throw new Error('Wallet not connected');
      return web3Provider.optimizePortfolio(sessionId, marketIds, chunkDuration);
    }, [connected, web3Provider]),
    
    getSession: useCallback(async (sessionId) => {
      if (!connected) throw new Error('Wallet not connected');
      return web3Provider.getSession(sessionId);
    }, [connected, web3Provider]),
    
    getSessionPositions: useCallback(async (sessionId) => {
      if (!connected) throw new Error('Wallet not connected');
      return web3Provider.getSessionPositions(sessionId);
    }, [connected, web3Provider]),
  };
}

export function useChainName() {
  const { chainId } = useWeb3();
  return web3Provider.getNetworkName(chainId);
}

export function useFormattedAddress() {
  const { account } = useWeb3();
  return account ? web3Provider.formatAddress(account) : null;
}