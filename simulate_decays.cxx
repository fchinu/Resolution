#include <TF1.h>
#include <TFile.h>
#include <TGenPhaseSpace.h>
#include <TH2D.h>
#include <TTree.h>
#include <TLorentzVector.h> // ugly, but TGenPhaseSpace uses this.
#include <TMath.h>
#include <TRandom3.h>
#include <TROOT.h>
#include <TString.h>

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

constexpr double kK0sMass = 0.497648;
constexpr double kPiMass = 0.1395703918;

// Store plain kinematics so the smearing loop never mutates shared data
struct DecayEvent {
  float pos_pt, pos_eta, pos_phi;
  float neg_pt, neg_eta, neg_phi;
};

void simulate_decays(char* name, int kNtrials, int seed, unsigned nThreads = 30){
  ROOT::EnableThreadSafety();

  TRandom3 seedRng(seed);

  TLorentzVector mother;
  TGenPhaseSpace gen2Pi;
  const double massesDau[2]{kPiMass, kPiMass};

  TF1 mtExpo("mtExpo","[0]*x*std::exp(-std::hypot([2], x)/[1])", 0.1, 20.);
  mtExpo.SetParameter(0, 1.);
  mtExpo.SetParameter(1, 0.5);
  mtExpo.SetParameter(2, kK0sMass);

  // --- Generate all decays once ---
  std::vector<DecayEvent> events(kNtrials);
  {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> genThreads;
    genThreads.reserve(nThreads);
    for (unsigned t = 0; t < nThreads; ++t) {
      genThreads.emplace_back([&, t]() {
        const size_t blockSize = kNtrials / nThreads;
        const size_t iStart    = t * blockSize;
        const size_t iEnd      = (t + 1 == nThreads) ? static_cast<size_t>(kNtrials) : iStart + blockSize;
        TRandom3 rng(static_cast<unsigned>(seed) + t);
        gRandom = &rng; // TGenPhaseSpace::Generate() uses gRandom internally
        TLorentzVector loc_mother;
        TGenPhaseSpace loc_gen;
        for (size_t i = iStart; i < iEnd; ++i) {
          const float pT_k0s = rng.Uniform(0.1, 20.); // mtExpo.GetRandom();
          const float eta    = rng.Uniform(-0.3, 0.3);
          const float phi    = rng.Uniform(0, TMath::TwoPi());
          loc_mother.SetPtEtaPhiM(pT_k0s, eta, phi, kK0sMass);
          loc_gen.SetDecay(loc_mother, 2, massesDau);
          loc_gen.Generate();
          const TLorentzVector* pos = loc_gen.GetDecay(0);
          const TLorentzVector* neg = loc_gen.GetDecay(1);
          events[i] = {
            (float)pos->Pt(), (float)pos->Eta(), (float)pos->Phi(),
            (float)neg->Pt(), (float)neg->Eta(), (float)neg->Phi()
          };
        }
      });
    }
    for (auto& th : genThreads) th.join();
    std::cout << "Generated " << kNtrials << " events in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
              << " s\n";
  }

  // Fill TTRee
  TFile fout(name, "RECREATE");
  TTree tree("decays", "K0s -> pi+ pi- kinematics");
  DecayEvent buf;
  tree.Branch("pos_pt",  &buf.pos_pt,  "pos_pt/F");
  tree.Branch("pos_eta", &buf.pos_eta, "pos_eta/F");
  tree.Branch("pos_phi", &buf.pos_phi, "pos_phi/F");
  tree.Branch("neg_pt",  &buf.neg_pt,  "neg_pt/F");
  tree.Branch("neg_eta", &buf.neg_eta, "neg_eta/F");
  tree.Branch("neg_phi", &buf.neg_phi, "neg_phi/F");

  const auto twrite = std::chrono::steady_clock::now();
  for (const auto& ev : events) {
    buf = ev;
    tree.Fill();
  }
  tree.Write();
  fout.Close();
  std::cout << "Wrote " << kNtrials << " events to " << name << "in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - twrite).count()
            << " s\n";
}