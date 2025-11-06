import { ethers } from 'ethers';
import Web3Modal from 'web3modal';
import WalletConnectProvider from '@walletconnect/web3-provider';
import { CONTRACTS } from './contracts';

// Web3Modal configuration
const providerOptions = {
  walletconnect: {
    package: WalletConnectProvider,
    options: {
      rpc: {
        1: process.env.REACT_APP_ETHEREUM_RPC_URL,
        137: process.env.REACT_APP_POLYGON_RPC_URL,
        42161: process.env.REACT_APP_ARBITRUM_RPC_URL,
      },
    },
  },
};

export class Web3Provider {
  constructor() {
    this.provider = null;
    this.signer = null;
    this.address = null;
    this.chainId = null;
    this.contracts = {};
    
    if (typeof window !== 'undefined') {
      this.web3Modal = new Web3Modal({
        cacheProvider: true,
        providerOptions,
        theme: {
          background: '#0a0a0a',
          main: '#00ff00',
          secondary: '#00ff00',
          border: '#222',
          hover: '#1a1a1a',
        },
      });
    }
  }

  async connect() {
    try {
      const instance = await this.web3Modal.connect();
      const provider = new ethers.providers.Web3Provider(instance);
      const signer = provider.getSigner();
      const address = await signer.getAddress();
      const network = await provider.getNetwork();
      
      this.provider = provider;
      this.signer = signer;
      this.address = address;
      this.chainId = network.chainId;
      
      // Initialize contracts
      this.initializeContracts();
      
      // Subscribe to accounts change
      instance.on('accountsChanged', (accounts) => {
        window.location.reload();
      });
      
      // Subscribe to chainId change
      instance.on('chainChanged', (chainId) => {
        window.location.reload();
      });
      
      return {
        provider: this.provider,
        signer: this.signer,
        address: this.address,
        chainId: this.chainId,
      };
    } catch (error) {
      console.error('Failed to connect wallet:', error);
      throw error;
    }
  }

  async disconnect() {
    if (this.web3Modal) {
      await this.web3Modal.clearCachedProvider();
      this.provider = null;
      this.signer = null;
      this.address = null;
      this.chainId = null;
      this.contracts = {};
    }
  }

  initializeContracts() {
    const networkContracts = CONTRACTS[this.chainId];
    if (!networkContracts) {
      throw new Error(`No contracts found for chainId ${this.chainId}`);
    }
    
    // Initialize trading engine contract
    this.contracts.tradingEngine = new ethers.Contract(
      networkContracts.tradingEngine.address,
      networkContracts.tradingEngine.abi,
      this.signer
    );
    
    // Initialize Kelly optimizer contract
    this.contracts.kellyOptimizer = new ethers.Contract(
      networkContracts.kellyOptimizer.address,
      networkContracts.kellyOptimizer.abi,
      this.signer
    );
    
    // Initialize chunk manager contract
    this.contracts.chunkManager = new ethers.Contract(
      networkContracts.chunkManager.address,
      networkContracts.chunkManager.abi,
      this.signer
    );
  }

  async createSession(initialBankroll) {
    if (!this.contracts.tradingEngine) {
      throw new Error('Trading engine not initialized');
    }
    
    const tx = await this.contracts.tradingEngine.createSession(
      ethers.utils.parseEther(initialBankroll.toString())
    );
    const receipt = await tx.wait();
    
    // Extract session ID from events
    const event = receipt.events.find(e => e.event === 'SessionCreated');
    return event.args.sessionId.toString();
  }

  async placeBet(sessionId, marketId, stake, outcome) {
    if (!this.contracts.tradingEngine) {
      throw new Error('Trading engine not initialized');
    }
    
    const tx = await this.contracts.tradingEngine.placeBet(
      sessionId,
      marketId,
      ethers.utils.parseEther(stake.toString()),
      outcome
    );
    const receipt = await tx.wait();
    
    // Extract position ID from events
    const event = receipt.events.find(e => e.event === 'PositionPlaced');
    return event.args.positionId.toString();
  }

  async optimizePortfolio(sessionId, marketIds, chunkDuration) {
    if (!this.contracts.tradingEngine) {
      throw new Error('Trading engine not initialized');
    }
    
    const result = await this.contracts.tradingEngine.optimizePortfolio(
      sessionId,
      marketIds,
      chunkDuration
    );
    
    return {
      stakes: result.stakes.map(s => ethers.utils.formatEther(s)),
      outcomes: result.outcomes,
    };
  }

  async getSession(sessionId) {
    if (!this.contracts.tradingEngine) {
      throw new Error('Trading engine not initialized');
    }
    
    const session = await this.contracts.tradingEngine.getSession(sessionId);
    
    return {
      id: session.id.toString(),
      trader: session.trader,
      initialBankroll: ethers.utils.formatEther(session.initialBankroll),
      currentBankroll: ethers.utils.formatEther(session.currentBankroll),
      startTime: new Date(session.startTime.toNumber() * 1000),
      lastActivityTime: new Date(session.lastActivityTime.toNumber() * 1000),
      isActive: session.isActive,
      totalBetsPlaced: session.totalBetsPlaced.toNumber(),
      totalBetsWon: session.totalBetsWon.toNumber(),
      totalProfit: ethers.utils.formatEther(session.totalProfit),
    };
  }

  async getSessionPositions(sessionId) {
    if (!this.contracts.tradingEngine) {
      throw new Error('Trading engine not initialized');
    }
    
    const positions = await this.contracts.tradingEngine.getSessionPositions(sessionId);
    
    return positions.map(pos => ({
      id: pos.id.toString(),
      sessionId: pos.sessionId.toString(),
      marketId: pos.marketId,
      stake: ethers.utils.formatEther(pos.stake),
      odds: parseFloat(ethers.utils.formatEther(pos.odds)),
      outcome: pos.outcome,
      timestamp: new Date(pos.timestamp.toNumber() * 1000),
      isSettled: pos.isSettled,
      isWon: pos.isWon,
      payout: ethers.utils.formatEther(pos.payout),
    }));
  }

  async switchNetwork(chainId) {
    if (!window.ethereum) {
      throw new Error('No ethereum provider found');
    }
    
    const chainIdHex = `0x${chainId.toString(16)}`;
    
    try {
      await window.ethereum.request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: chainIdHex }],
      });
    } catch (error) {
      // This error code indicates that the chain has not been added to MetaMask
      if (error.code === 4902) {
        const networkParams = this.getNetworkParams(chainId);
        await window.ethereum.request({
          method: 'wallet_addEthereumChain',
          params: [networkParams],
        });
      } else {
        throw error;
      }
    }
  }

  getNetworkParams(chainId) {
    const networks = {
      137: {
        chainId: '0x89',
        chainName: 'Polygon',
        nativeCurrency: {
          name: 'MATIC',
          symbol: 'MATIC',
          decimals: 18,
        },
        rpcUrls: [process.env.REACT_APP_POLYGON_RPC_URL],
        blockExplorerUrls: ['https://polygonscan.com/'],
      },
      42161: {
        chainId: '0xa4b1',
        chainName: 'Arbitrum One',
        nativeCurrency: {
          name: 'ETH',
          symbol: 'ETH',
          decimals: 18,
        },
        rpcUrls: [process.env.REACT_APP_ARBITRUM_RPC_URL],
        blockExplorerUrls: ['https://arbiscan.io/'],
      },
    };
    
    return networks[chainId];
  }

  formatAddress(address) {
    if (!address) return '';
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  }

  getNetworkName(chainId) {
    const networks = {
      1: 'Ethereum',
      137: 'Polygon',
      42161: 'Arbitrum',
      11155111: 'Sepolia',
      80001: 'Mumbai',
      421613: 'Arbitrum Goerli',
    };
    
    return networks[chainId] || 'Unknown';
  }
}

// Export singleton instance
export const web3Provider = new Web3Provider();