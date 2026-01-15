"""
QGNN Base Model with Simulated Noise Support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pennylane as qml
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

#=====================================
# DATA LOADING
#=====================================

class MolecularDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        self.molecules = []
        self.gaps = []
        self.mol_ids = []
        
        for item in data_list:
            mol = Chem.MolFromSmiles(item['smiles'])
            if mol is not None:
                self.molecules.append(mol)
                self.gaps.append(item['gap'])
                self.mol_ids.append(item.get('mol_id', 'unknown'))
        
        self.gaps = torch.FloatTensor(self.gaps)
        print(f"Dataset ready with {len(self.molecules)} molecules")
    
    def __len__(self):
        return len(self.molecules)
    
    def __getitem__(self, idx):
        return self.molecules[idx], self.gaps[idx]

def custom_collate_fn(batch):
    molecules = []
    gaps = []
    for mol, gap in batch:
        molecules.append(mol)
        gaps.append(gap)
    return molecules, torch.stack(gaps)

def load_molecular_data(csv_file, max_molecules=2000):
    print(f"Loading data from {csv_file}...")
    
    df = pd.read_csv(csv_file)
    print(f"Found {len(df)} molecules in file")
    
    # Check format and convert if needed
    if 'gap' in df.columns and df['gap'].iloc[0] < 1.0:
        print("Converting from Hartree to eV...")
        HARTREE_TO_EV = 27.2114
        gap_values = df['gap'] * HARTREE_TO_EV
    else:
        gap_values = df['gap']
    
    valid_data = []
    skipped = 0
    
    for idx, row in df.iterrows():
        if len(valid_data) >= max_molecules:
            break
            
        try:
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol and mol.GetNumHeavyAtoms() <= 9:
                valid_data.append({
                    'mol_id': row.get('mol_id', f'mol_{idx}'),
                    'smiles': row['smiles'],
                    'gap': float(gap_values[idx]),
                    'n_atoms': mol.GetNumHeavyAtoms()
                })
            else:
                skipped += 1
        except:
            skipped += 1
            continue
    
    print(f"Loaded {len(valid_data)} valid molecules, skipped {skipped}")
    
    gaps = [d['gap'] for d in valid_data]
    print(f"Gap range: [{min(gaps):.3f}, {max(gaps):.3f}] eV")
    print(f"Average gap: {np.mean(gaps):.3f} +/- {np.std(gaps):.3f} eV")
    
    return valid_data

#=====================================
# EDU-QGC MODEL WITH NOISE
#=====================================

class EDU_QGC_Layer(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        
        # Improved initialization
        self.node_params = nn.Parameter(torch.randn(n_qubits, 3) * 0.05)
        self.bond_params = nn.Parameter(torch.randn(4, 2) * 0.05)
        self.diagonal_params = nn.Parameter(torch.randn(4) * 0.05)
        
        # Additional phase correction
        self.phase_params = nn.Parameter(torch.zeros(4))
    
    def apply_node_layer(self, n_atoms):
        for i in range(n_atoms):
            qml.Rot(self.node_params[i, 0],
                   self.node_params[i, 1],
                   self.node_params[i, 2],
                   wires=i)
    
    def apply_edu(self, i, j, bond_type):
        bond_map = {1.0: 0, 1.5: 1, 2.0: 2, 3.0: 3}
        idx = bond_map.get(bond_type, 0)
        
        # V operation
        qml.RY(self.bond_params[idx, 0], wires=j)
        qml.RZ(self.bond_params[idx, 1], wires=j)
        
        # D operation
        qml.CNOT(wires=[i, j])
        qml.RZ(self.diagonal_params[idx], wires=j)
        qml.CNOT(wires=[i, j])
        
        # Phase correction
        qml.PhaseShift(self.phase_params[idx], wires=j)
        
        # V† operation
        qml.RZ(-self.bond_params[idx, 1], wires=j)
        qml.RY(-self.bond_params[idx, 0], wires=j)
    
    def apply_link_layer(self, bond_list):
        bonds_by_type = {1.0: [], 1.5: [], 2.0: [], 3.0: []}
        
        for i, j, bond_type in bond_list:
            if bond_type in bonds_by_type:
                bonds_by_type[bond_type].append((i, j))
        
        for bond_type in [1.0, 1.5, 2.0, 3.0]:
            for i, j in bonds_by_type[bond_type]:
                self.apply_edu(i, j, bond_type)

class QGNN_Model(nn.Module):
    """QGNN Model with Simulated Noise Support"""
    def __init__(self, n_qubits=12, n_layers=3, use_master=True, noise_level=0.0):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_master = use_master
        self.noise_level = noise_level  # Simulated noise level
        
        if use_master:
            self.n_atom_qubits = n_qubits - 1
            self.master_idx = n_qubits - 1
        else:
            self.n_atom_qubits = n_qubits
            self.master_idx = None
        
        self.max_atoms = min(self.n_atom_qubits, 9)
        
        # Use Lightning simulator
        self.dev = qml.device('lightning.qubit', wires=n_qubits)
        
        # Improved atom encoding with better initialization
        self.atom_params = nn.ParameterDict({
            'C': nn.Parameter(torch.tensor([0.1, 0.1])),
            'N': nn.Parameter(torch.tensor([0.15, 0.1])),
            'O': nn.Parameter(torch.tensor([0.2, 0.1])),
            'F': nn.Parameter(torch.tensor([0.25, 0.1])),
            'S': nn.Parameter(torch.tensor([0.3, 0.1])),
            'Cl': nn.Parameter(torch.tensor([0.35, 0.1]))
        })
        
        # EDU layers
        self.edu_layers = nn.ModuleList([
            EDU_QGC_Layer(n_qubits) for _ in range(n_layers)
        ])
        
        # Master node with attention-like mechanism
        if use_master:
            self.master_params = nn.Parameter(torch.randn(n_layers, 3) * 0.05)
            self.master_attention = nn.Parameter(torch.ones(n_layers, self.n_atom_qubits) * 0.1)
        
        # Improved output network with dropout
        self.output_net = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Define quantum circuit
        self.qnode = qml.QNode(self.quantum_circuit, self.dev, 
                              interface="torch", diff_method="parameter-shift")
    
    def get_molecular_info(self, mol):
        atoms = []
        atom_features = []
        
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() > 1:
                atoms.append(atom)
                atom_features.append({
                    'symbol': atom.GetSymbol(),
                    'degree': atom.GetDegree(),
                    'aromatic': atom.GetIsAromatic(),
                    'hybridization': str(atom.GetHybridization())
                })
        
        bond_list = []
        atom_indices = {atom.GetIdx(): i for i, atom in enumerate(atoms)}
        
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            
            if begin in atom_indices and end in atom_indices:
                i = atom_indices[begin]
                j = atom_indices[end]
                
                bond_type = bond.GetBondTypeAsDouble()
                if bond.GetIsAromatic():
                    bond_type = 1.5
                
                if i < self.max_atoms and j < self.max_atoms:
                    bond_list.append((i, j, bond_type))
        
        return atom_features[:self.max_atoms], bond_list
    
    def quantum_circuit(self, mol_data):
        atoms = mol_data['atoms']
        bonds = mol_data['bonds']
        n_atoms = len(atoms)
        
        # Enhanced atom encoding
        for i, atom in enumerate(atoms):
            symbol = atom['symbol']
            if symbol in self.atom_params:
                params = self.atom_params[symbol]
                
                # Include more molecular features
                degree_factor = atom['degree'] / 4.0
                aromatic_factor = 0.5 if atom['aromatic'] else 0.0
                
                qml.RY(params[0] * (1 + degree_factor), wires=i)
                qml.RZ(params[1] * (1 + aromatic_factor), wires=i)
        
        # Initialize master node
        if self.use_master:
            qml.Hadamard(wires=self.master_idx)
        
        # Apply layers with residual-like connections
        for layer in range(self.n_layers):
            # Node update
            self.edu_layers[layer].apply_node_layer(n_atoms)
            
            # Bond update
            if bonds:
                self.edu_layers[layer].apply_link_layer(bonds)
            
            # Enhanced master node interaction
            if self.use_master and n_atoms > 0:
                # Weighted connections based on attention
                for i in range(n_atoms):
                    weight = torch.sigmoid(self.master_attention[layer, i])
                    qml.CRY(weight * 2, wires=[i, self.master_idx])
                
                # Master evolution
                qml.Rot(self.master_params[layer, 0],
                       self.master_params[layer, 1],
                       self.master_params[layer, 2],
                       wires=self.master_idx)
        
        # Measure all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def apply_simulated_noise(self, quantum_output):
        """Apply simulated noise effect based on gate count"""
        if self.noise_level == 0:
            return quantum_output
        
        # Estimate total gate count (approximately)
        gates_per_layer = 25  # Rough estimate: node ops + bond ops
        total_gates = self.n_layers * gates_per_layer
        
        # Calculate cumulative error probability
        # Noise accumulates with circuit depth
        error_prob = 1 - (1 - self.noise_level) ** total_gates
        
        # Apply noise effect
        signal_attenuation = 1 - error_prob
        noise_magnitude = error_prob * 0.2  # Scale factor for noise
        
        # Add Gaussian noise and attenuate signal
        noise = torch.randn_like(quantum_output) * noise_magnitude
        noisy_output = quantum_output * signal_attenuation + noise
        
        return noisy_output
    
    def forward(self, mol):
        atoms, bonds = self.get_molecular_info(mol)
        
        if len(atoms) == 0:
            return torch.tensor(0.0)
        
        mol_data = {'atoms': atoms, 'bonds': bonds}
        quantum_out = self.qnode(mol_data)
        quantum_out = torch.stack(quantum_out).float()
        
        # Apply simulated noise
        if self.noise_level > 0:
            quantum_out = self.apply_simulated_noise(quantum_out)
        
        output = self.output_net(quantum_out)
        return output.squeeze()

#=====================================
# TRAINING FUNCTIONS
#=====================================

def train_one_epoch(model, loader, optimizer, device, gradient_clip=1.0):
    model.train()
    total_loss = 0
    n_batches = 0
    
    progress = tqdm(loader, desc="Training")
    for molecules, targets in progress:
        batch_loss = 0
        batch_size = 0
        
        for mol, target in zip(molecules, targets):
            try:
                optimizer.zero_grad()
                
                pred = model(mol)
                loss = nn.MSELoss()(pred, target)
                
                loss.backward()
                
                # Gradient clipping
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                optimizer.step()
                
                batch_loss += loss.item()
                batch_size += 1
                
            except Exception as e:
                continue
        
        if batch_size > 0:
            avg_loss = batch_loss / batch_size
            total_loss += avg_loss
            n_batches += 1
            progress.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / n_batches if n_batches > 0 else float('inf')

def evaluate_model(model, loader, device):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for molecules, batch_targets in tqdm(loader, desc="Evaluating"):
            for mol, target in zip(molecules, batch_targets):
                try:
                    pred = model(mol)
                    predictions.append(pred.item())
                    targets.append(target.item())
                except:
                    continue
    
    if len(predictions) == 0:
        return float('inf'), float('inf'), 0.0, np.array([]), np.array([])
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    
    return mae, rmse, r2, predictions, targets

def train_qgnn_model(config, output_dir=None):
    """Main training function that returns results"""
    
    # Set random seeds
    np.random.seed(config.get('seed', 42))
    torch.manual_seed(config.get('seed', 42))
    
    # Create output directory
    if output_dir is None:
        output_dir = f"qgnn_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    import json
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load data
    print("Loading data...")
    data = load_molecular_data(config['csv_file'], config['max_molecules'])
    
    # Split data
    train_val, test_data = train_test_split(data, test_size=config['test_split'], 
                                           random_state=config.get('seed', 42))
    train_data, val_data = train_test_split(train_val, test_size=config['val_split'], 
                                           random_state=config.get('seed', 42))
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create datasets and loaders
    train_dataset = MolecularDataset(train_data)
    val_dataset = MolecularDataset(val_data)
    test_dataset = MolecularDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                            shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                          shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                           shuffle=False, collate_fn=custom_collate_fn)
    
    # Create model with noise if specified
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QGNN_Model(
        n_qubits=config['n_qubits'],
        n_layers=config['n_layers'],
        use_master=config['use_master'],
        noise_level=config.get('noise_level', 0.0)  # Add noise level
    ).to(device)
    
    print(f"Model created: {config['n_layers']} layers, noise_level={config.get('noise_level', 0.0)}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['learning_rate'],
                          weight_decay=config['weight_decay'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=config['lr_patience'],
        factor=config['lr_factor'], min_lr=config['lr_min']
    )
    
    # Training loop
    train_losses = []
    val_metrics = []
    best_val_mae = float('inf')
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(1, config['n_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['n_epochs']}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, 
                                   config['gradient_clip'])
        train_losses.append(train_loss)
        
        # Validate
        val_mae, val_rmse, val_r2, _, _ = evaluate_model(model, val_loader, device)
        val_metrics.append({'mae': val_mae, 'rmse': val_rmse, 'r2': val_r2})
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R2: {val_r2:.4f}")
        
        # Learning rate scheduling
        if config['lr_schedule']:
            scheduler.step(val_mae)
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if config['early_stopping'] and patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Save checkpoint
        if epoch % config.get('save_every_epoch', 10) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_metrics': val_metrics
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    # Test final model
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    test_mae, test_rmse, test_r2, test_preds, test_targets = evaluate_model(
        model, test_loader, device
    )
    
    print(f"\nTest Results:")
    print(f"MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")
    
    # Save results
    results = {
        'config': config,
        'train_losses': train_losses,
        'val_metrics': val_metrics,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'epochs_trained': len(train_losses),
        'best_val_mae': best_val_mae,
        'output_dir': output_dir,
        'noise_level': config.get('noise_level', 0.0)
    }
    
    # Save detailed results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create plots
    create_plots(train_losses, val_metrics, test_preds, test_targets, output_dir)
    
    return results

def create_plots(train_losses, val_metrics, test_preds, test_targets, output_dir):
    """Create and save plots"""
    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    val_maes = [m['mae'] for m in val_metrics]
    ax2.plot(epochs, val_maes, 'g-', label='Validation MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE (eV)')
    ax2.set_title('Validation Performance')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    # Predictions scatter plot
    if len(test_preds) > 0:
        plt.figure(figsize=(8, 8))
        plt.scatter(test_targets, test_preds, alpha=0.6, s=50)
        
        min_val = min(test_targets.min(), test_preds.min())
        max_val = max(test_targets.max(), test_preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('True HOMO-LUMO Gap (eV)')
        plt.ylabel('Predicted HOMO-LUMO Gap (eV)')
        plt.title('Predictions vs Ground Truth')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'predictions.png'), dpi=150)
        plt.close()

        