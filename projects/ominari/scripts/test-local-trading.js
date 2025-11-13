const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("🎮 Ominari DApp - Local Trading Test");
  console.log("=====================================\n");
  
  // Load deployment
  const deploymentPath = path.join(__dirname, "../deployments/localhost.json");
  const deployment = JSON.parse(fs.readFileSync(deploymentPath, "utf-8"));
  
  // Get signers
  const [deployer, oracle, alice, bob] = await ethers.getSigners();
  
  console.log("👥 Test Accounts:");
  console.log("  Alice:", alice.address);
  console.log("  Bob:", bob.address);
  console.log();
  
  // Connect to contracts
  const tradingEngine = await ethers.getContractAt(
    "OminariTradingEngine",
    deployment.contracts.tradingEngine
  );
  
  // 1. Create sessions for both users
  console.log("📝 Creating trading sessions...");
  
  const aliceSessionTx = await tradingEngine.connect(alice).createSession(
    ethers.utils.parseEther("10.0") // 10 ETH bankroll
  );
  const aliceReceipt = await aliceSessionTx.wait();
  const aliceSessionId = aliceReceipt.events.find(e => e.event === "SessionCreated").args.sessionId;
  console.log("  ✓ Alice's session:", aliceSessionId.toString());
  
  const bobSessionTx = await tradingEngine.connect(bob).createSession(
    ethers.utils.parseEther("5.0") // 5 ETH bankroll
  );
  const bobReceipt = await bobSessionTx.wait();
  const bobSessionId = bobReceipt.events.find(e => e.event === "SessionCreated").args.sessionId;
  console.log("  ✓ Bob's session:", bobSessionId.toString());
  
  // 2. Show available markets
  console.log("\n🏈 Available Markets:");
  for (const market of deployment.markets) {
    const marketId = ethers.utils.formatBytes32String(market.id);
    const marketData = await tradingEngine.markets(marketId);
    
    console.log(`\n  ${marketData.homeTeam} vs ${marketData.awayTeam}`);
    console.log(`  Market ID: ${market.id}`);
    console.log(`  Odds: Home ${ethers.utils.formatEther(marketData.odds[0])} | Draw ${ethers.utils.formatEther(marketData.odds[1])} | Away ${ethers.utils.formatEther(marketData.odds[2])}`);
    
    const maturityDate = new Date(marketData.maturityDate.toNumber() * 1000);
    console.log(`  Match time: ${maturityDate.toLocaleString()}`);
  }
  
  // 3. Run Kelly optimization for Alice
  console.log("\n📊 Running Kelly Optimization for Alice...");
  const marketIds = deployment.markets.map(m => ethers.utils.formatBytes32String(m.id));
  
  const optimizeTx = await tradingEngine.connect(alice).optimizePortfolio(
    aliceSessionId,
    marketIds,
    120 // 2 hour chunks
  );
  const optimizeReceipt = await optimizeTx.wait();
  
  const optimizeEvent = optimizeReceipt.events.find(e => e.event === "PortfolioOptimized");
  if (optimizeEvent) {
    console.log(`  ✓ Optimization complete: ${optimizeEvent.args.marketCount} markets analyzed`);
  }
  
  // Get recommended stakes from return value
  // Note: In production, this would be done off-chain first
  const result = await tradingEngine.connect(alice).callStatic.optimizePortfolio(
    aliceSessionId,
    marketIds,
    120
  );
  
  console.log(`  Recommendations: ${result.stakes.length} positions`);
  
  // 4. Place some bets
  console.log("\n🎲 Placing bets...");
  
  // Alice places a bet
  const aliceBetTx = await tradingEngine.connect(alice).placeBet(
    aliceSessionId,
    ethers.utils.formatBytes32String(deployment.markets[0].id),
    ethers.utils.parseEther("0.5"), // 0.5 ETH stake
    0 // Home win
  );
  const aliceBetReceipt = await aliceBetTx.wait();
  const alicePositionId = aliceBetReceipt.events.find(e => e.event === "PositionPlaced").args.positionId;
  console.log(`  ✓ Alice bet 0.5 ETH on ${deployment.markets[0].home} (Home) - Position #${alicePositionId}`);
  
  // Bob places a bet
  const bobBetTx = await tradingEngine.connect(bob).placeBet(
    bobSessionId,
    ethers.utils.formatBytes32String(deployment.markets[1].id),
    ethers.utils.parseEther("1.0"), // 1.0 ETH stake
    2 // Away win
  );
  const bobBetReceipt = await bobBetTx.wait();
  const bobPositionId = bobBetReceipt.events.find(e => e.event === "PositionPlaced").args.positionId;
  console.log(`  ✓ Bob bet 1.0 ETH on ${deployment.markets[1].away} (Away) - Position #${bobPositionId}`);
  
  // 5. Check session status
  console.log("\n📈 Session Status:");
  
  const aliceSession = await tradingEngine.getSession(aliceSessionId);
  console.log(`\n  Alice's Session:`);
  console.log(`    Initial bankroll: ${ethers.utils.formatEther(aliceSession.initialBankroll)} ETH`);
  console.log(`    Current bankroll: ${ethers.utils.formatEther(aliceSession.currentBankroll)} ETH`);
  console.log(`    Bets placed: ${aliceSession.totalBetsPlaced}`);
  
  const bobSession = await tradingEngine.getSession(bobSessionId);
  console.log(`\n  Bob's Session:`);
  console.log(`    Initial bankroll: ${ethers.utils.formatEther(bobSession.initialBankroll)} ETH`);
  console.log(`    Current bankroll: ${ethers.utils.formatEther(bobSession.currentBankroll)} ETH`);
  console.log(`    Bets placed: ${bobSession.totalBetsPlaced}`);
  
  // 6. Show all positions
  console.log("\n📋 Active Positions:");
  
  const alicePositions = await tradingEngine.getSessionPositions(aliceSessionId);
  console.log(`\n  Alice's positions (${alicePositions.length}):`);
  for (const pos of alicePositions) {
    const market = await tradingEngine.markets(pos.marketId);
    console.log(`    - ${ethers.utils.formatEther(pos.stake)} ETH on ${['Home', 'Draw', 'Away'][pos.outcome]}`);
    console.log(`      Odds: ${ethers.utils.formatEther(pos.odds)} | Potential payout: ${ethers.utils.formatEther(pos.stake.mul(pos.odds).div(ethers.constants.WeiPerEther))} ETH`);
  }
  
  const bobPositions = await tradingEngine.getSessionPositions(bobSessionId);
  console.log(`\n  Bob's positions (${bobPositions.length}):`);
  for (const pos of bobPositions) {
    const market = await tradingEngine.markets(pos.marketId);
    console.log(`    - ${ethers.utils.formatEther(pos.stake)} ETH on ${['Home', 'Draw', 'Away'][pos.outcome]}`);
    console.log(`      Odds: ${ethers.utils.formatEther(pos.odds)} | Potential payout: ${ethers.utils.formatEther(pos.stake.mul(pos.odds).div(ethers.constants.WeiPerEther))} ETH`);
  }
  
  // 7. Simulate match results (as oracle)
  console.log("\n⚽ Simulating match results...");
  console.log("  (In production, this would come from Chainlink oracles)");
  
  // Resolve first market - Home wins
  await tradingEngine.connect(oracle).resolveMarket(
    ethers.utils.formatBytes32String(deployment.markets[0].id),
    0 // Home wins
  );
  console.log(`  ✓ ${deployment.markets[0].home} wins!`);
  
  // Settle Alice's position
  await tradingEngine.connect(oracle).settlePosition(alicePositionId);
  console.log(`  ✓ Alice's position settled - WON!`);
  
  // Check Alice's updated session
  const aliceSessionFinal = await tradingEngine.getSession(aliceSessionId);
  console.log(`    Alice's new bankroll: ${ethers.utils.formatEther(aliceSessionFinal.currentBankroll)} ETH`);
  console.log(`    Alice's profit: ${ethers.utils.formatEther(aliceSessionFinal.totalProfit)} ETH`);
  
  console.log("\n✨ Test complete!");
  console.log("\n💡 Next steps:");
  console.log("  1. Start the frontend: npm run frontend:dev");
  console.log("  2. Connect MetaMask to localhost:8545");
  console.log("  3. Import test accounts using private keys from Hardhat");
  console.log("  4. Interact with the DApp through the UI!");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });