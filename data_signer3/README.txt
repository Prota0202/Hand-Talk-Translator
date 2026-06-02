Signeur 3 — adulte (cross-signeur)
==================================

Qui : une 3e personne (pas toi, pas le signeur 2 / petit frere).
Le modele n'est PAS reentraine : on mesure si ton LSTM signeur 1
generalise chez quelqu'un d'autre.

Protocole (6 glosses, sans alphabet) :
  MOI, NOM, Bonjour, 2, 4, ETUDIANT
  5 echantillons par gloss | mode AUTO (~1 s par signe)

--- Preparation du dossier ---

  .\scripts\prepare_signer3.ps1

  Nouvelle personne / tout effacer et recommencer :
  .\scripts\prepare_signer3.ps1 -Fresh

--- Collecte ---

  .\scripts\collect_signer3.ps1

  Ou :
  python collect_data.py --data-dir data_signer3 --signer3 --resume --samples 5 --auto

--- Evaluation ---

  .\scripts\eval_cross_signers.ps1

  Rapports : models\cross_signer_report.md
