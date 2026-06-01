Signeur 4 (cross-signeur)
=========================

4e personne testee avec le modele entraine sur tes donnees (signeur 1).
Pas de reentrainement : mesure de generalisation seulement.

Protocole (6 glosses, sans alphabet) — identique au signeur 3 :
  MOI, NOM, Bonjour, 2, 4, ETUDIANT
  5 echantillons par gloss | mode AUTO

Note le profil dans ton rapport (ex. ami, collegue, pratiquant LSF…).

--- Preparation ---

  .\scripts\prepare_signer4.ps1

  Recommencer a zero :
  .\scripts\prepare_signer4.ps1 -Fresh

--- Collecte ---

  .\scripts\collect_signer4.ps1

  Ou :
  python collect_data.py --data-dir data_signer4 --signer4 --resume --samples 5 --auto

--- Evaluation ---

  .\scripts\eval_cross_signers.ps1

  Detail signeur 4 : models\cross_signer_4.md
