Signeur 5 (cross-signeur)
=========================

5e personne testee avec le modele entraine sur tes donnees (signeur 1).
Pas de reentrainement.

Ami qui parle et pratique un peu la LSF : meilleure execution gestuelle
que les signeurs 3--4 (gestes appris sur le moment), d'ou le 100 % sur
le sous-ensemble teste (6 glosses).

Protocole (6 glosses, sans alphabet) :
  MOI, NOM, Bonjour, 2, 4, ETUDIANT
  5 echantillons par gloss | mode AUTO

--- Preparation ---

  .\scripts\prepare_signer5.ps1

  Recommencer a zero :
  .\scripts\prepare_signer5.ps1 -Fresh

--- Collecte ---

  .\scripts\collect_signer5.ps1

  Ou :
  python collect_data.py --data-dir data_signer5 --signer5 --resume --samples 5 --auto

--- Evaluation ---

  .\scripts\eval_cross_signers.ps1

  Detail : models\cross_signer_5.md
