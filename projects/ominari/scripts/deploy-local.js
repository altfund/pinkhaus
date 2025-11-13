const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  console.log("🚀 Starting local deployment...");
  
  // Get signers
  const [deployer, oracle] = await hre.ethers.getSigners();
  console.log("Deployer address:", deployer.address);
  console.log("Oracle address:", oracle.address);
  
  // Deploy Kelly Optimizer
  console.log("\n📊 Deploying KellyOptimizer...");
  const KellyOptimizer = await hre.ethers.getContractFactory("KellyOptimizer");
  const kellyOptimizer = await KellyOptimizer.deploy();
  await kellyOptimizer.deployed();
  console.log("KellyOptimizer deployed to:", kellyOptimizer.address);
  
  // Deploy Chunk Manager
  console.log("\n📦 Deploying ChunkManager...");
  const ChunkManager = await hre.ethers.getContractFactory("ChunkManager");
  const chunkManager = await ChunkManager.deploy();
  await chunkManager.deployed();
  console.log("ChunkManager deployed to:", chunkManager.address);
  
  // Deploy Trading Engine
  console.log("\n🔧 Deploying OminariTradingEngine...");
  const OminariTradingEngine = await hre.ethers.getContractFactory("OminariTradingEngine");
  const tradingEngine = await OminariTradingEngine.deploy(
    kellyOptimizer.address,
    chunkManager.address
  );
  await tradingEngine.deployed();
  console.log("OminariTradingEngine deployed to:", tradingEngine.address);
  
  // Set oracle address
  console.log("\n⚙️ Setting oracle address...");
  await tradingEngine.setOracle(oracle.address);
  console.log("Oracle set to:", oracle.address);
  
  // Add some test markets
  console.log("\n🏈 Adding test markets...");
  const markets = [
    {
      id: hre.ethers.utils.formatBytes32String("match-1"),
      home: "Manchester United",
      away: "Liverpool",
      odds: [
        hre.ethers.utils.parseEther("2.50"),
        hre.ethers.utils.parseEther("3.20"),
        hre.ethers.utils.parseEther("2.80")
      ],
      maturity: Math.floor(Date.now() / 1000) + 7200 // 2 hours from now
    },
    {
      id: hre.ethers.utils.formatBytes32String("match-2"),
      home: "Real Madrid",
      away: "Barcelona",
      odds: [
        hre.ethers.utils.parseEther("2.10"),
        hre.ethers.utils.parseEther("3.50"),
        hre.ethers.utils.parseEther("3.20")
      ],
      maturity: Math.floor(Date.now() / 1000) + 10800 // 3 hours from now
    }
  ];
  
  for (const market of markets) {
    await tradingEngine.connect(oracle).updateMarket(
      market.id,
      market.home,
      market.away,
      market.odds,
      market.maturity
    );
    console.log(`Added market: ${market.home} vs ${market.away}`);
  }
  
  // Save deployment addresses
  const deploymentData = {
    network: "localhost",
    chainId: 31337,
    deployer: deployer.address,
    oracle: oracle.address,
    contracts: {
      kellyOptimizer: kellyOptimizer.address,
      chunkManager: chunkManager.address,
      tradingEngine: tradingEngine.address
    },
    markets: markets.map(m => ({
      ...m,
      id: m.id.replace(/\0/g, '').trim()
    })),
    timestamp: new Date().toISOString()
  };
  
  // Save to file
  const deploymentsDir = path.join(__dirname, "../deployments");
  if (!fs.existsSync(deploymentsDir)) {
    fs.mkdirSync(deploymentsDir);
  }
  
  fs.writeFileSync(
    path.join(deploymentsDir, "localhost.json"),
    JSON.stringify(deploymentData, null, 2)
  );
  
  // Update .env.local with addresses
  const envPath = path.join(__dirname, "../.env.local");
  let envContent = fs.readFileSync(envPath, "utf-8");
  
  envContent = envContent.replace(
    /TRADING_ENGINE_ADDRESS=.*/,
    `TRADING_ENGINE_ADDRESS=${tradingEngine.address}`
  );
  envContent = envContent.replace(
    /KELLY_OPTIMIZER_ADDRESS=.*/,
    `KELLY_OPTIMIZER_ADDRESS=${kellyOptimizer.address}`
  );
  envContent = envContent.replace(
    /CHUNK_MANAGER_ADDRESS=.*/,
    `CHUNK_MANAGER_ADDRESS=${chunkManager.address}`
  );
  
  fs.writeFileSync(envPath, envContent);
  
  console.log("\n✅ Deployment complete!");
  console.log("\n📝 Contract addresses saved to:");
  console.log("  - deployments/localhost.json");
  console.log("  - .env.local");
  
  console.log("\n🎮 Test accounts (from Hardhat):");
  console.log("  Deployer:", deployer.address);
  console.log("  Oracle:", oracle.address);
  
  console.log("\n💰 To interact with contracts:");
  console.log("  1. Keep this terminal running");
  console.log("  2. In another terminal: npm run test:local");
  console.log("  3. Or use: npm run console:local");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });