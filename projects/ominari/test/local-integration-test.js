const { expect } = require("chai");
const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

describe("Ominari DApp - Local Integration Test", function() {
  let tradingEngine, kellyOptimizer, chunkManager;
  let deployer, oracle, user1, user2;
  let deployment;
  
  before(async function() {
    // Load deployment data
    const deploymentPath = path.join(__dirname, "../deployments/localhost.json");
    if (fs.existsSync(deploymentPath)) {
      deployment = JSON.parse(fs.readFileSync(deploymentPath, "utf-8"));
      console.log("📋 Loaded deployment from:", deploymentPath);
    }
    
    // Get signers
    [deployer, oracle, user1, user2] = await ethers.getSigners();
    
    // Get deployed contracts
    if (deployment) {
      tradingEngine = await ethers.getContractAt(
        "OminariTradingEngine",
        deployment.contracts.tradingEngine
      );
      kellyOptimizer = await ethers.getContractAt(
        "KellyOptimizer",
        deployment.contracts.kellyOptimizer
      );
      chunkManager = await ethers.getContractAt(
        "ChunkManager",
        deployment.contracts.chunkManager
      );
      console.log("✅ Connected to deployed contracts");
    } else {
      console.log("⚠️  No deployment found, deploying new contracts...");
      // Deploy new contracts if needed
      const KellyOptimizer = await ethers.getContractFactory("KellyOptimizer");
      kellyOptimizer = await KellyOptimizer.deploy();
      
      const ChunkManager = await ethers.getContractFactory("ChunkManager");
      chunkManager = await ChunkManager.deploy();
      
      const OminariTradingEngine = await ethers.getContractFactory("OminariTradingEngine");
      tradingEngine = await OminariTradingEngine.deploy(
        kellyOptimizer.address,
        chunkManager.address
      );
      
      await tradingEngine.setOracle(oracle.address);
    }
  });
  
  describe("1. Session Management", function() {
    it("Should create a trading session", async function() {
      const initialBankroll = ethers.utils.parseEther("1.0");
      
      const tx = await tradingEngine.connect(user1).createSession(initialBankroll);
      const receipt = await tx.wait();
      
      // Find SessionCreated event
      const event = receipt.events.find(e => e.event === "SessionCreated");
      expect(event).to.not.be.undefined;
      
      const sessionId = event.args.sessionId;
      console.log("   ✓ Created session:", sessionId.toString());
      
      // Verify session data
      const session = await tradingEngine.getSession(sessionId);
      expect(session.trader).to.equal(user1.address);
      expect(session.initialBankroll).to.equal(initialBankroll);
      expect(session.currentBankroll).to.equal(initialBankroll);
      expect(session.isActive).to.be.true;
    });
    
    it("Should reject session with insufficient bankroll", async function() {
      const lowBankroll = ethers.utils.parseEther("0.005"); // Below minimum
      
      await expect(
        tradingEngine.connect(user2).createSession(lowBankroll)
      ).to.be.revertedWith("Insufficient initial bankroll");
    });
  });
  
  describe("2. Market Data", function() {
    it("Should have test markets available", async function() {
      if (!deployment || !deployment.markets) {
        console.log("   ⚠️  No test markets in deployment");
        return;
      }
      
      for (const market of deployment.markets) {
        const marketId = ethers.utils.formatBytes32String(market.id);
        const marketData = await tradingEngine.markets(marketId);
        
        console.log(`   ✓ Market: ${marketData.homeTeam} vs ${marketData.awayTeam}`);
        console.log(`     Odds: ${ethers.utils.formatEther(marketData.odds[0])}, ${ethers.utils.formatEther(marketData.odds[1])}, ${ethers.utils.formatEther(marketData.odds[2])}`);
      }
    });
  });
  
  describe("3. Kelly Optimization", function() {
    it("Should calculate Kelly fraction correctly", async function() {
      const odds = ethers.utils.parseEther("2.5");
      const probability = ethers.utils.parseEther("0.5"); // 50%
      
      const kellyFraction = await tradingEngine.calculateKellyFraction(
        ethers.utils.parseEther("100"),
        odds,
        probability
      );
      
      console.log("   ✓ Kelly fraction:", ethers.utils.formatEther(kellyFraction));
      expect(kellyFraction).to.be.gt(0);
    });
    
    it("Should optimize portfolio with multiple markets", async function() {
      // Skip if no session created
      if (!deployment) return;
      
      const sessionId = 1; // From first test
      const marketIds = deployment.markets.map(m => ethers.utils.formatBytes32String(m.id));
      
      const result = await tradingEngine.optimizePortfolio(
        sessionId,
        marketIds,
        120 // 2 hour chunks
      );
      
      console.log("   ✓ Optimization complete");
      console.log(`     Recommended stakes: ${result.stakes.length}`);
    });
  });
  
  describe("4. Position Management", function() {
    let sessionId;
    let positionId;
    
    before(async function() {
      // Create a session for position tests
      const tx = await tradingEngine.connect(user1).createSession(
        ethers.utils.parseEther("1.0")
      );
      const receipt = await tx.wait();
      const event = receipt.events.find(e => e.event === "SessionCreated");
      sessionId = event.args.sessionId;
    });
    
    it("Should place a bet", async function() {
      if (!deployment || !deployment.markets || deployment.markets.length === 0) {
        console.log("   ⚠️  No markets available for betting");
        return;
      }
      
      const market = deployment.markets[0];
      const marketId = ethers.utils.formatBytes32String(market.id);
      const stake = ethers.utils.parseEther("0.1");
      const outcome = 0; // Home win
      
      const tx = await tradingEngine.connect(user1).placeBet(
        sessionId,
        marketId,
        stake,
        outcome
      );
      const receipt = await tx.wait();
      
      const event = receipt.events.find(e => e.event === "PositionPlaced");
      positionId = event.args.positionId;
      
      console.log("   ✓ Placed bet:", positionId.toString());
      console.log(`     Stake: ${ethers.utils.formatEther(stake)} ETH`);
      console.log(`     Outcome: ${outcome} (Home)`);
      
      // Verify position
      const position = await tradingEngine.getPosition(positionId);
      expect(position.stake).to.equal(stake);
      expect(position.outcome).to.equal(outcome);
      expect(position.isSettled).to.be.false;
      
      // Verify session bankroll updated
      const session = await tradingEngine.getSession(sessionId);
      expect(session.currentBankroll).to.equal(
        ethers.utils.parseEther("0.9") // 1.0 - 0.1
      );
    });
    
    it("Should get all positions for a session", async function() {
      const positions = await tradingEngine.getSessionPositions(sessionId);
      console.log(`   ✓ Found ${positions.length} position(s) in session`);
      expect(positions.length).to.be.gte(1);
    });
  });
  
  describe("5. Chunk Management", function() {
    it("Should create market chunks", async function() {
      const marketIds = [
        ethers.utils.formatBytes32String("match-1"),
        ethers.utils.formatBytes32String("match-2"),
        ethers.utils.formatBytes32String("match-3")
      ];
      
      const now = Math.floor(Date.now() / 1000);
      const maturityDates = [
        now + 3600,   // 1 hour
        now + 7200,   // 2 hours
        now + 14400   // 4 hours
      ];
      
      const sports = ["soccer", "soccer", "soccer"];
      
      const params = {
        minChunkDuration: 60,
        maxChunkDuration: 240,
        settlementBuffer: 15,
        maxMarketsPerChunk: 5,
        targetCapitalPerChunk: ethers.utils.parseEther("0.5")
      };
      
      const chunkIds = await chunkManager.createChunks(
        marketIds,
        maturityDates,
        sports,
        params
      );
      
      console.log(`   ✓ Created ${chunkIds.length} chunk(s)`);
    });
    
    it("Should get sport timings", async function() {
      const soccerTimings = await chunkManager.getSportTimings("soccer");
      
      console.log("   ✓ Soccer timings:");
      console.log(`     Average duration: ${soccerTimings.averageDuration} minutes`);
      console.log(`     Settlement time: ${soccerTimings.settlementTime} minutes`);
      
      expect(soccerTimings.averageDuration).to.be.gt(0);
    });
  });
  
  describe("6. Security Tests", function() {
    it("Should only allow oracle to update markets", async function() {
      const marketId = ethers.utils.formatBytes32String("test-market");
      
      await expect(
        tradingEngine.connect(user1).updateMarket(
          marketId,
          "Team A",
          "Team B",
          [ethers.utils.parseEther("2"), ethers.utils.parseEther("3"), ethers.utils.parseEther("4")],
          Math.floor(Date.now() / 1000) + 3600
        )
      ).to.be.revertedWith("Only oracle can call");
    });
    
    it("Should only allow session owner to place bets", async function() {
      const sessionId = 1; // From earlier test
      const marketId = ethers.utils.formatBytes32String("match-1");
      
      await expect(
        tradingEngine.connect(user2).placeBet(
          sessionId,
          marketId,
          ethers.utils.parseEther("0.1"),
          0
        )
      ).to.be.revertedWith("Not session owner");
    });
    
    it("Should allow owner to pause contract", async function() {
      await tradingEngine.connect(deployer).pause();
      
      await expect(
        tradingEngine.connect(user1).createSession(ethers.utils.parseEther("1"))
      ).to.be.revertedWith("Pausable: paused");
      
      await tradingEngine.connect(deployer).unpause();
      console.log("   ✓ Pause/unpause functionality working");
    });
  });
});

// Run specific test
if (process.argv.includes("--quick")) {
  describe.only("Quick Smoke Test", function() {
    it("Should connect to contracts", async function() {
      const deploymentPath = path.join(__dirname, "../deployments/localhost.json");
      expect(fs.existsSync(deploymentPath)).to.be.true;
      console.log("✅ Deployment file exists");
    });
  });
}